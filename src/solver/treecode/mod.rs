// why: several support methods (`apply`, `traverse`,
// `build_expansions`) exist only for cross-checking the multipole +
// octree plumbing under `#[cfg(test)]`; the cartesian `Expansion`
// also carries unused Laplace evaluators that will disappear once
// Yukawa migrates to SH. Allow rather than individually gate each.
#![allow(dead_code)]

//! Barnes-Hut treecode that drives [`BemOperator::apply`]. Two
//! independent walks share one octree: the Laplace walk uses the
//! P = 6 spherical-harmonic expansion in [`solid_harmonic`] at a
//! loose MAC, the Yukawa walk uses the cartesian Taylor-order-2
//! expansion in [`multipole`] at a tighter MAC (its accuracy ceiling
//! limits MAC widening — see `solver::operator`).
//!
//! ```text
//! top[a] = ½ f_a + Σ_b K'₀(a,b) f_b − ε_ratio · Σ_b K₀(a,b) h_b
//! bot[a] = ½ f_a − Σ_b K'_κ(a,b) f_b + Σ_b K_κ(a,b) h_b
//! ```
//!
//! Near-field pairs fall through to the panel-integrated quadratures
//! in [`crate::solver::kernel`]; far-field clusters are evaluated via
//! the multipole expansions. A point-source [`PointTreecode::apply`]
//! entry is kept test-only as a cross-check harness against direct
//! summation.

mod multipole;
mod octree;
mod solid_harmonic;

use crate::geometry::panel::FaceGeoms;
use crate::solver::kernel::{block_entries_laplace, block_entries_yukawa};
#[cfg(test)]
use crate::solver::panel_integrals::FOUR_PI;
#[cfg(test)]
use glam::DVec3;
use multipole::Expansion;
use octree::Tree;
use rayon::prelude::*;
use solid_harmonic::ShExpansion;

// why: traversal stack depth is bounded by the octree's max recursion
// depth times the branching factor (each internal node can push up to
// 8 children onto the stack before we pop any of them). `octree.rs`
// caps depth at 24, so 8·24 = 192 — a hair of headroom over the
// theoretical worst case, reached only on pathological meshes.
const TRAVERSE_STACK_CAPACITY: usize = 8 * 24;

/// Point-source treecode over face centroids. Single octree but two
/// Barnes-Hut MAC thresholds: a loose one for the Laplace kernels
/// (driven by the `P = 6` SH expansion, `θ^7` error envelope) and a
/// tighter one for the Yukawa kernels (driven by the cartesian
/// Taylor-order-2 expansion, `θ²` error envelope). Two independent
/// traversals share the tree topology but test their own MAC at each
/// internal node, so the Laplace walk sees far more far-field
/// acceptance and correspondingly fewer near-field pair evaluations.
#[derive(Debug)]
pub(super) struct PointTreecode<'g> {
    geom: &'g FaceGeoms,
    tree: Tree,
    theta_laplace: f64,
    theta_yukawa: f64,
}

impl<'g> PointTreecode<'g> {
    /// Build the tree. `theta_laplace` / `theta_yukawa` are the two
    /// Barnes-Hut MAC parameters (bounding-sphere radius over
    /// distance); `n_crit` is the max panels per leaf.
    pub(super) fn new(
        geom: &'g FaceGeoms,
        theta_laplace: f64,
        theta_yukawa: f64,
        n_crit: usize,
    ) -> Self {
        let tree = Tree::new(&geom.centroids, n_crit);
        Self {
            geom,
            tree,
            theta_laplace,
            theta_yukawa,
        }
    }

    /// Evaluate `Σ_{b ≠ a} q_b · G(c_a, c_b)` at every centroid, with
    /// `G = G₀` when `kappa == 0` and `G = G_κ` otherwise.
    /// Production callers want [`Self::apply_bem_operator`]; this
    /// pure-point-source entry stays as a cross-check target for the
    /// octree + multipole plumbing. The MAC parameter picked matches
    /// the kernel: Laplace uses `theta_laplace`, Yukawa uses
    /// `theta_yukawa`.
    #[cfg(test)]
    pub(super) fn apply(&self, q: &[f64], kappa: f64) -> Vec<f64> {
        debug_assert_eq!(q.len(), self.geom.len());
        debug_assert!(kappa >= 0.0, "kappa must be non-negative");
        let expansions = self.build_expansions(q);
        let theta = if kappa == 0.0 {
            self.theta_laplace
        } else {
            self.theta_yukawa
        };
        // why: `map_init` gives each rayon worker its own scratch
        // stack, reused across all target panels that worker handles.
        // Replaces a per-target `Vec::with_capacity(...)` that would
        // otherwise malloc/free ~N × n_iter times per solve.
        (0..self.geom.len())
            .into_par_iter()
            .map_init(
                || Vec::<usize>::with_capacity(TRAVERSE_STACK_CAPACITY),
                |stack, a| {
                    stack.clear();
                    self.traverse(self.geom.centroids[a], q, kappa, theta, &expansions, stack)
                },
            )
            .collect()
    }

    /// Apply the full Juffer BIE block operator via the treecode.
    ///
    /// Returns `(top, bot)` where
    ///
    /// ```text
    /// top[a] = ½ f_a + Σ_b K'₀(a,b) f_b − ε_ratio · Σ_b K₀(a,b) h_b
    /// bot[a] = ½ f_a − Σ_b K'_κ(a,b) f_b + Σ_b K_κ(a,b) h_b
    /// ```
    ///
    /// Runs two independent tree walks per target: the Laplace walk
    /// at `theta_laplace` drives the top block (and, when `κ = 0`,
    /// the bottom block too via Yukawa ≡ Laplace); the Yukawa walk
    /// at `theta_yukawa` drives the bottom block when screening is
    /// active. Near-field uses the panel-integrated quadratures;
    /// far-field uses the SH Laplace expansion + cartesian Yukawa
    /// expansion respectively.
    pub(super) fn apply_bem_operator(
        &self,
        x_f: &[f64],
        x_h: &[f64],
        kappa: f64,
        eps_ratio: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.geom.len();
        debug_assert_eq!(x_f.len(), n);
        debug_assert_eq!(x_h.len(), n);
        debug_assert!(kappa >= 0.0, "kappa must be non-negative");

        let (sh_exps, cart_exps) = self.build_bem_expansions(x_f, x_h, kappa);

        (0..n)
            .into_par_iter()
            .map_init(
                || Vec::<usize>::with_capacity(TRAVERSE_STACK_CAPACITY),
                |stack, a| {
                    let half_f = 0.5 * x_f[a];
                    stack.clear();
                    let (k0_h, k0p_f) = self.traverse_laplace(a, x_f, x_h, &sh_exps, stack);
                    let top = half_f + k0p_f - eps_ratio * k0_h;
                    let bot = if kappa == 0.0 {
                        // κ = 0: Yukawa ≡ Laplace, reuse the sums.
                        half_f + k0_h - k0p_f
                    } else {
                        stack.clear();
                        let (kk_h, kkp_f) =
                            self.traverse_yukawa(a, x_f, x_h, kappa, &cart_exps, stack);
                        half_f + kk_h - kkp_f
                    };
                    (top, bot)
                },
            )
            .unzip()
    }

    /// Build BEM moments. The SH store always carries both Laplace
    /// single- and double-layer moments. The cartesian store is empty
    /// when `κ = 0` (Yukawa traversal is skipped; reusing the Laplace
    /// values) and populated otherwise for the Yukawa pair.
    ///
    /// why: when screening is active, the two sweeps are fully
    /// independent (disjoint output buffers, all inputs immutable);
    /// `rayon::join` halves the build-phase wallclock on multicore
    /// machines without any data hazards to reason about.
    fn build_bem_expansions(
        &self,
        x_f: &[f64],
        x_h: &[f64],
        kappa: f64,
    ) -> (Vec<ShExpansion>, Vec<Expansion>) {
        let geom = self.geom;
        let sh_leaf = |exp: &mut ShExpansion, pid: usize| {
            let s = geom.centroids[pid];
            let area = geom.areas[pid];
            exp.accumulate(s, x_h[pid] * area);
            exp.accumulate_dipole(s, x_f[pid] * area, geom.normals[pid]);
        };
        if kappa == 0.0 {
            (self.build_sh_expansions_with(sh_leaf), Vec::new())
        } else {
            rayon::join(
                || self.build_sh_expansions_with(sh_leaf),
                || {
                    self.build_expansions_with(|exp, pid| {
                        let s = geom.centroids[pid];
                        let area = geom.areas[pid];
                        exp.accumulate(s, x_h[pid] * area);
                        exp.accumulate_dipole(s, x_f[pid] * area, geom.normals[pid]);
                    })
                },
            )
        }
    }

    /// Laplace-only traversal at `theta_laplace`. Returns the pair
    /// `(Σ_b K_0(a,b) · h_b, Σ_b K'_0(a,b) · f_b)` — the caller
    /// composes these into the `top` Juffer row (and, at `κ = 0`,
    /// the `bot` row too).
    ///
    /// Near-field uses the Laplace-only [`block_entries_laplace`],
    /// far-field uses [`ShExpansion::evaluate_laplace_pair`] which
    /// returns both single and double in one irregular-harmonic pass.
    fn traverse_laplace(
        &self,
        target_a: usize,
        x_f: &[f64],
        x_h: &[f64],
        sh_exps: &[ShExpansion],
        stack: &mut Vec<usize>,
    ) -> (f64, f64) {
        let target = self.geom.centroids[target_a];
        let mut k0_sum = 0.0;
        let mut k0p_sum = 0.0;
        stack.push(0);
        while let Some(idx) = stack.pop() {
            let node = &self.tree.nodes[idx];
            if node.is_leaf() {
                for &pid in &node.panel_ids {
                    let b = pid as usize;
                    let (k0, k0p) = block_entries_laplace(self.geom, target_a, b);
                    k0_sum += k0 * x_h[b];
                    k0p_sum += k0p * x_f[b];
                }
                continue;
            }
            let dist = (target - node.bbox.center).length();
            if node.bbox.bounding_radius() < self.theta_laplace * dist {
                let (k0_far, k0p_far) = sh_exps[idx].evaluate_laplace_pair(target);
                k0_sum += k0_far;
                k0p_sum += k0p_far;
            } else {
                for &ci in &node.children {
                    stack.push(ci as usize);
                }
            }
        }
        (k0_sum, k0p_sum)
    }

    /// Yukawa-only traversal at `theta_yukawa`. Returns the pair
    /// `(Σ_b K_κ(a,b) · h_b, Σ_b K'_κ(a,b) · f_b)`; the caller
    /// composes these into the `bot` Juffer row. Debug-asserts
    /// `κ > 0`; at `κ = 0` the caller reuses the Laplace sums via the
    /// `K_κ ≡ K_0` identity.
    fn traverse_yukawa(
        &self,
        target_a: usize,
        x_f: &[f64],
        x_h: &[f64],
        kappa: f64,
        cart_exps: &[Expansion],
        stack: &mut Vec<usize>,
    ) -> (f64, f64) {
        debug_assert!(kappa > 0.0, "traverse_yukawa requires κ > 0");
        let target = self.geom.centroids[target_a];
        let mut kk_sum = 0.0;
        let mut kkp_sum = 0.0;
        stack.push(0);
        while let Some(idx) = stack.pop() {
            let node = &self.tree.nodes[idx];
            if node.is_leaf() {
                for &pid in &node.panel_ids {
                    let b = pid as usize;
                    let (kk, kkp) = block_entries_yukawa(self.geom, target_a, b, kappa);
                    kk_sum += kk * x_h[b];
                    kkp_sum += kkp * x_f[b];
                }
                continue;
            }
            let dist = (target - node.bbox.center).length();
            if node.bbox.bounding_radius() < self.theta_yukawa * dist {
                let (kk_far, kkp_far) = cart_exps[idx].evaluate_yukawa_pair(target, kappa);
                kk_sum += kk_far;
                kkp_sum += kkp_far;
            } else {
                for &ci in &node.children {
                    stack.push(ci as usize);
                }
            }
        }
        (kk_sum, kkp_sum)
    }

    /// Point-source single-layer upward sweep. Test-only; production
    /// uses [`Self::build_bem_expansions`].
    #[cfg(test)]
    fn build_expansions(&self, q: &[f64]) -> Vec<Expansion> {
        let geom = self.geom;
        self.build_expansions_with(|exp, pid| {
            exp.accumulate(geom.centroids[pid], q[pid]);
        })
    }

    /// Spherical-harmonic upward sweep. Same reverse-order post-order
    /// traversal as [`Self::build_expansions_with`], but over the SH
    /// coefficient store.
    fn build_sh_expansions_with<F>(&self, mut leaf_accum: F) -> Vec<ShExpansion>
    where
        F: FnMut(&mut ShExpansion, usize),
    {
        let mut exps: Vec<ShExpansion> = self
            .tree
            .nodes
            .iter()
            .map(|node| ShExpansion {
                center: node.bbox.center,
                ..ShExpansion::default()
            })
            .collect();
        for i in (0..self.tree.nodes.len()).rev() {
            let node = &self.tree.nodes[i];
            if node.is_leaf() {
                for &pid in &node.panel_ids {
                    leaf_accum(&mut exps[i], pid as usize);
                }
            } else {
                let parent_center = node.bbox.center;
                let children = node.children.clone();
                for ci in children {
                    let shifted = exps[ci as usize].shifted_to(parent_center);
                    exps[i].fold(&shifted);
                }
            }
        }
        exps
    }

    /// Upward sweep skeleton shared by every moment-build entry point.
    /// `leaf_accum(exp, panel_id)` is called once per panel inside its
    /// leaf's scope; internal nodes gather children via
    /// [`Expansion::shifted_to`] + [`Expansion::fold`].
    ///
    /// why: `Tree::new` assigns child indices strictly greater than
    /// parent, so iterating the node array in reverse gives a natural
    /// post-order — each internal node finds its children's moments
    /// already populated.
    fn build_expansions_with<F>(&self, mut leaf_accum: F) -> Vec<Expansion>
    where
        F: FnMut(&mut Expansion, usize),
    {
        let mut exps: Vec<Expansion> = self
            .tree
            .nodes
            .iter()
            .map(|node| Expansion {
                center: node.bbox.center,
                ..Expansion::default()
            })
            .collect();
        for i in (0..self.tree.nodes.len()).rev() {
            let node = &self.tree.nodes[i];
            if node.is_leaf() {
                for &pid in &node.panel_ids {
                    leaf_accum(&mut exps[i], pid as usize);
                }
            } else {
                let parent_center = node.bbox.center;
                // why: snapshot children before folding so the borrow
                // checker is happy writing `exps[i]` while reading
                // `exps[child]`. Vec<u32> × ≤ 8 entries; negligible.
                let children = node.children.clone();
                for ci in children {
                    let shifted = exps[ci as usize].shifted_to(parent_center);
                    exps[i].fold(&shifted);
                }
            }
        }
        exps
    }

    /// Iterative traversal from the root; no recursion into Rust's
    /// call stack (tree depth can be up to 24 on pathological meshes).
    /// `stack` is caller-supplied scratch — reused across targets by
    /// the outer `map_init` loop.
    #[cfg(test)]
    fn traverse(
        &self,
        target: DVec3,
        q: &[f64],
        kappa: f64,
        theta: f64,
        exps: &[Expansion],
        stack: &mut Vec<usize>,
    ) -> f64 {
        let mut acc = 0.0;
        stack.push(0);
        while let Some(idx) = stack.pop() {
            let node = &self.tree.nodes[idx];
            if node.is_leaf() {
                // Direct summation over the leaf's panels. Skip the
                // singular r → 0 case (target centroid equals a source
                // centroid — i.e. the self-interaction). The κ = 0
                // branch avoids a pure-waste `exp(0)` call on Laplace
                // solves.
                if kappa == 0.0 {
                    for &pid in &node.panel_ids {
                        let r = (target - self.geom.centroids[pid as usize]).length();
                        if r > 1e-14 {
                            acc += q[pid as usize] / (FOUR_PI * r);
                        }
                    }
                } else {
                    for &pid in &node.panel_ids {
                        let r = (target - self.geom.centroids[pid as usize]).length();
                        if r > 1e-14 {
                            acc += q[pid as usize] * (-kappa * r).exp() / (FOUR_PI * r);
                        }
                    }
                }
                continue;
            }
            // why: only internal nodes need `dist` (for the MAC test
            // and nowhere else). Computing it unconditionally above
            // the is_leaf branch wastes one `sqrt` per leaf visit —
            // ~O(N/n_crit) per target per apply, which across a
            // full lysozyme solve was measured at tens of millions
            // of wasted `sqrt` calls.
            let dist = (target - node.bbox.center).length();
            if node.bbox.bounding_radius() < theta * dist {
                acc += if kappa == 0.0 {
                    exps[idx].evaluate_laplace(target)
                } else {
                    exps[idx].evaluate_yukawa(target, kappa)
                };
            } else {
                for &ci in &node.children {
                    stack.push(ci as usize);
                }
            }
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Surface;
    use crate::solver::kernel::block_entries;

    /// Hand-rolled N² summation for cross-validation. `kappa == 0`
    /// gives the Laplace kernel, `> 0` gives Yukawa.
    fn direct_sum(centroids: &[DVec3], q: &[f64], kappa: f64) -> Vec<f64> {
        let n = centroids.len();
        (0..n)
            .map(|a| {
                let mut s = 0.0;
                for b in 0..n {
                    if a == b {
                        continue;
                    }
                    let r = (centroids[a] - centroids[b]).length();
                    let kernel = if kappa == 0.0 {
                        1.0
                    } else {
                        (-kappa * r).exp()
                    };
                    s += q[b] * kernel / (FOUR_PI * r);
                }
                s
            })
            .collect()
    }

    #[test]
    fn treecode_matches_direct_on_sphere_smooth_rhs() {
        // 320-face icosphere (subdiv=3) — heterogeneous enough that
        // many internal nodes pass the MAC, non-degenerate enough
        // that every centroid has neighbours at a range of distances.
        //
        // Taylor-order-2 is only accurate for smooth / positive-
        // dominated RHS; alternating-sign vectors produce near-
        // cancellation in the true potential and relative error
        // blows up to ~30 %. The GMRES vectors we'll feed to the
        // treecode in production are Krylov-smoothed surface
        // densities — closer to smooth than to alternating — so
        // we validate against the smooth case here and raise the
        // expansion order later if production meshes disagree.
        let surface = Surface::icosphere(10.0, 3);
        let geom = surface.geom_internal();
        let treecode = PointTreecode::new(geom, 0.5, 0.5, 8);

        let n = geom.len();
        let smooth_rhs: Vec<Vec<f64>> = vec![
            vec![1.0; n],
            (0..n)
                .map(|i| (i as f64 * 0.3).sin() + 0.5 * (i as f64 * 0.11).cos() + 1.0)
                .collect(),
        ];

        for (k, q) in smooth_rhs.iter().enumerate() {
            let got = treecode.apply(q, 0.0);
            let exact = direct_sum(&geom.centroids, q, 0.0);
            let max_rel: f64 = got
                .iter()
                .zip(&exact)
                .map(|(&g, &e)| {
                    if e.abs() > 1e-12 {
                        (g - e).abs() / e.abs()
                    } else {
                        (g - e).abs()
                    }
                })
                .fold(0.0, f64::max);
            assert!(
                max_rel < 0.01,
                "rhs {k}: treecode vs direct max rel err {max_rel:e}"
            );
        }
    }

    #[test]
    fn treecode_recovers_direct_at_theta_zero() {
        // θ = 0 means the MAC is never satisfied, so every node
        // recurses all the way to leaves: the treecode reduces to
        // direct summation. This catches any leaf-visit accounting
        // bug (missed panel, double-count, etc.).
        let surface = Surface::icosphere(5.0, 2);
        let geom = surface.geom_internal();
        let treecode = PointTreecode::new(geom, 0.0, 0.0, 4);
        let n = geom.len();
        let q: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
        let got = treecode.apply(&q, 0.0);
        let exact = direct_sum(&geom.centroids, &q, 0.0);
        for (i, (&g, &e)) in got.iter().zip(&exact).enumerate() {
            assert!(
                (g - e).abs() < 1e-12 * (1.0 + e.abs()),
                "entry {i}: treecode={g}, direct={e}"
            );
        }
    }

    #[test]
    fn treecode_laplace_equals_yukawa_at_kappa_zero() {
        // κ = 0 must route through evaluate_laplace; we double-check
        // that evaluate_yukawa-with-κ=0 would have given the same
        // answer (the code path is split but the math is identical).
        let surface = Surface::icosphere(5.0, 2);
        let geom = surface.geom_internal();
        let treecode = PointTreecode::new(geom, 0.3, 0.3, 4);
        let n = geom.len();
        let q: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
        let laplace = treecode.apply(&q, 0.0);
        let yukawa_eps = treecode.apply(&q, 1e-12);
        for (i, (&lap, &yuk)) in laplace.iter().zip(&yukawa_eps).enumerate() {
            assert!(
                (lap - yuk).abs() < 1e-9 * (1.0 + lap.abs()),
                "entry {i}: laplace={lap}, yukawa(κ→0)={yuk}"
            );
        }
    }

    #[test]
    fn treecode_yukawa_matches_direct_on_sphere() {
        // Physiological κ = 1/8 Å⁻¹ on a 10 Å sphere: κR ≈ 1.25 at
        // panel-to-panel distances, so the exponential attenuation
        // is non-trivial but not extreme.
        let surface = Surface::icosphere(10.0, 3);
        let geom = surface.geom_internal();
        let kappa = 0.125;
        let treecode = PointTreecode::new(geom, 0.5, 0.5, 8);

        let n = geom.len();
        let q: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.3).sin() + 0.5 * (i as f64 * 0.11).cos() + 1.0)
            .collect();

        let got = treecode.apply(&q, kappa);
        let exact = direct_sum(&geom.centroids, &q, kappa);
        let max_rel: f64 = got
            .iter()
            .zip(&exact)
            .map(|(&g, &e)| {
                if e.abs() > 1e-12 {
                    (g - e).abs() / e.abs()
                } else {
                    (g - e).abs()
                }
            })
            .fold(0.0, f64::max);
        // Yukawa decay makes the series behave even better than
        // Laplace (far contributions attenuate), so we reuse the
        // Laplace tolerance. 1 % is well inside Taylor-order-2's
        // expected envelope at θ = 0.5.
        assert!(
            max_rel < 0.01,
            "yukawa treecode vs direct max rel err {max_rel:e}"
        );
    }

    /// Direct, panel-integrated equivalent of
    /// [`PointTreecode::apply_bem_operator`] for comparison —
    /// explicit O(N²) double loop over `block_entries`.
    fn direct_bem_apply(
        geom: &FaceGeoms,
        x_f: &[f64],
        x_h: &[f64],
        kappa: f64,
        eps_ratio: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = geom.len();
        let mut top = vec![0.0_f64; n];
        let mut bot = vec![0.0_f64; n];
        for a in 0..n {
            let mut t = 0.5 * x_f[a];
            let mut b_acc = 0.5 * x_f[a];
            for b in 0..n {
                let (k0, k0p, kk, kkp) = block_entries(geom, a, b, kappa);
                t += k0p * x_f[b] - eps_ratio * k0 * x_h[b];
                b_acc += -kkp * x_f[b] + kk * x_h[b];
            }
            top[a] = t;
            bot[a] = b_acc;
        }
        (top, bot)
    }

    #[test]
    fn apply_bem_operator_near_field_matches_direct() {
        // With N_CRIT large enough that every panel lives in the root
        // leaf, the treecode's apply_bem_operator is pure leaf
        // direct-summation over block_entries — no multipole path
        // exercised. Must match the direct O(N²) loop to fp roundoff.
        let surface = Surface::icosphere(5.0, 2);
        let geom = surface.geom_internal();
        let n = geom.len();
        let treecode = PointTreecode::new(geom, 0.5, 0.5, n + 1);
        let x_f: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let x_h: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17).cos()).collect();
        let kappa = 0.125;
        let eps_ratio = 20.0;
        let (got_top, got_bot) = treecode.apply_bem_operator(&x_f, &x_h, kappa, eps_ratio);
        let (want_top, want_bot) = direct_bem_apply(geom, &x_f, &x_h, kappa, eps_ratio);
        for (i, (&g, &w)) in got_top.iter().zip(&want_top).enumerate() {
            assert!(
                (g - w).abs() < 1e-12 * (1.0 + w.abs()),
                "top[{i}]: treecode={g}, direct={w}"
            );
        }
        for (i, (&g, &w)) in got_bot.iter().zip(&want_bot).enumerate() {
            assert!(
                (g - w).abs() < 1e-12 * (1.0 + w.abs()),
                "bot[{i}]: treecode={g}, direct={w}"
            );
        }
    }

    #[test]
    fn treecode_yukawa_screening_reduces_magnitude() {
        // For a positive-charge cluster, the Yukawa sum must be
        // strictly smaller (in magnitude) than the Laplace sum at
        // every target — exp(−κR) ≤ 1. Sanity check against basic
        // sign / scaling bugs in evaluate_yukawa.
        let surface = Surface::icosphere(10.0, 2);
        let geom = surface.geom_internal();
        let treecode = PointTreecode::new(geom, 0.4, 0.4, 8);
        let n = geom.len();
        let q = vec![1.0; n];
        let laplace = treecode.apply(&q, 0.0);
        let yukawa = treecode.apply(&q, 0.2);
        for (i, (&lap, &yuk)) in laplace.iter().zip(&yukawa).enumerate() {
            assert!(
                yuk > 0.0 && yuk < lap,
                "entry {i}: lap={lap}, yuk={yuk} — expected 0 < yuk < lap"
            );
        }
    }
}
