// why: the whole treecode sub-module is Stage-1 scaffolding — its
// integration into `BemOperator::apply` lands in a later commit.
// Until then everything here appears dead to the linter even though
// the unit tests exercise it fully.
#![allow(dead_code)]

//! Barnes-Hut treecode for the Laplace single-layer point summation.
//!
//! Stage 1 scaffolding: an octree of panel centroids + Taylor-order-2
//! multipole expansions, evaluating
//!
//! ```text
//! Σ_b  q_b / (4π |c_a − c_b|)
//! ```
//!
//! at every target centroid `c_a`. This is the *point-source* Laplace
//! single-layer summation — no panel integration, no double-layer, no
//! Yukawa. It stresses the octree + multipole + MAC infrastructure
//! and gives us a cross-check against the hand-rolled N² direct sum.
//!
//! Stage 2 will add the Yukawa multipole and stage 3 will wire a
//! panel-aware variant into `BemOperator::apply`.

mod multipole_laplace;
mod octree;

use crate::geometry::panel::FaceGeoms;
use crate::solver::panel_integrals::FOUR_PI;
use glam::DVec3;
use multipole_laplace::Expansion;
use octree::Tree;
use rayon::prelude::*;

// why: traversal stack depth is bounded by the octree's max recursion
// depth times the branching factor (each internal node can push up to
// 8 children onto the stack before we pop any of them). `octree.rs`
// caps depth at 24, so 8·24 = 192 — a hair of headroom over the
// theoretical worst case, reached only on pathological meshes.
const TRAVERSE_STACK_CAPACITY: usize = 8 * 24;

/// Point-source Laplace treecode. Owns a tree of face centroids and
/// evaluates `Σ q_b / (4π |c_a − c_b|)` at every centroid per call.
#[derive(Debug)]
pub(super) struct LaplacePointTreecode<'g> {
    geom: &'g FaceGeoms,
    tree: Tree,
    theta: f64,
}

impl<'g> LaplacePointTreecode<'g> {
    /// Build the tree. `theta` is the Barnes-Hut MAC parameter
    /// (bounding-sphere radius over distance); `n_crit` is the max
    /// panels per leaf.
    pub(super) fn new(geom: &'g FaceGeoms, theta: f64, n_crit: usize) -> Self {
        let tree = Tree::new(&geom.centroids, n_crit);
        Self { geom, tree, theta }
    }

    /// Evaluate the point-source sum. `q.len()` must equal the panel
    /// count. Returns a fresh `Vec<f64>` of length `q.len()`, target
    /// index `a` holding `Σ_{b ≠ a} q_b / (4π |c_a − c_b|)`.
    pub(super) fn apply(&self, q: &[f64]) -> Vec<f64> {
        debug_assert_eq!(q.len(), self.geom.len());
        let expansions = self.build_expansions(q);
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
                    self.traverse(self.geom.centroids[a], q, &expansions, stack)
                },
            )
            .collect()
    }

    /// Upward sweep: compute a multipole expansion per tree node.
    /// Because `Tree::new` assigns children indices strictly greater
    /// than their parent, iterating the node array in reverse gives
    /// a natural post-order: by the time we reach an internal node
    /// all of its children already have expansions computed.
    fn build_expansions(&self, q: &[f64]) -> Vec<Expansion> {
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
                    exps[i].accumulate(self.geom.centroids[pid as usize], q[pid as usize]);
                }
            } else {
                let parent_center = node.bbox.center;
                // Child-fold; we snapshot each child before folding so
                // the borrow checker is happy writing back into
                // `exps[i]` while reading `exps[child]`.
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
    fn traverse(
        &self,
        target: DVec3,
        q: &[f64],
        exps: &[Expansion],
        stack: &mut Vec<usize>,
    ) -> f64 {
        let mut acc = 0.0;
        stack.push(0);
        while let Some(idx) = stack.pop() {
            let node = &self.tree.nodes[idx];
            let d = target - node.bbox.center;
            let dist = d.length();
            if node.is_leaf() {
                // Direct summation over the leaf's panels. Skip the
                // singular r → 0 case (target centroid equals a source
                // centroid — i.e. the self-interaction).
                for &pid in &node.panel_ids {
                    let r = (target - self.geom.centroids[pid as usize]).length();
                    if r > 1e-14 {
                        acc += q[pid as usize] / (FOUR_PI * r);
                    }
                }
                continue;
            }
            if node.bbox.bounding_radius() < self.theta * dist {
                acc += exps[idx].evaluate(target);
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

    /// Hand-rolled N² point-source summation for cross-validation.
    fn direct_point_sum(centroids: &[DVec3], q: &[f64]) -> Vec<f64> {
        let n = centroids.len();
        (0..n)
            .map(|a| {
                let mut s = 0.0;
                for b in 0..n {
                    if a == b {
                        continue;
                    }
                    let r = (centroids[a] - centroids[b]).length();
                    s += q[b] / (FOUR_PI * r);
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
        let treecode = LaplacePointTreecode::new(geom, 0.5, 8);

        let n = geom.len();
        let smooth_rhs: Vec<Vec<f64>> = vec![
            vec![1.0; n],
            (0..n)
                .map(|i| (i as f64 * 0.3).sin() + 0.5 * (i as f64 * 0.11).cos() + 1.0)
                .collect(),
        ];

        for (k, q) in smooth_rhs.iter().enumerate() {
            let got = treecode.apply(q);
            let exact = direct_point_sum(&geom.centroids, q);
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
        let treecode = LaplacePointTreecode::new(geom, 0.0, 4);
        let n = geom.len();
        let q: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
        let got = treecode.apply(&q);
        let exact = direct_point_sum(&geom.centroids, &q);
        for (i, (&g, &e)) in got.iter().zip(&exact).enumerate() {
            assert!(
                (g - e).abs() < 1e-12 * (1.0 + e.abs()),
                "entry {i}: treecode={g}, direct={e}"
            );
        }
    }
}
