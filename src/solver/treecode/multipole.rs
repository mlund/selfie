//! Cartesian Taylor-order-2 multipole for the Laplace and Yukawa
//! kernels. Expands around a cluster centre `c`:
//!
//! ```text
//! G(r, c + ε) ≈ G(r, c)
//!             + ε · ∇_s G(r, s)|_{s=c}
//!             + ½ ε ⊗ ε : ∇∇_s G(r, s)|_{s=c}
//!             + O(|ε|³)
//! ```
//!
//! The moments — monopole `Q`, dipole `p`, quadrupole `Q_ij` — are
//! kernel-agnostic sums over sources. Only the per-node evaluation
//! differs: [`Expansion::evaluate_laplace`] uses derivatives of
//! `G₀(r, s) = 1/(4π|r − s|)`; [`Expansion::evaluate_yukawa`] uses
//! derivatives of `G_κ = exp(−κ|r − s|)/(4π|r − s|)`. At κ = 0 the
//! Yukawa form reduces identically to the Laplace form.
//!
//! Relative truncation error is `O((|ε|/|r−c|)³)`; with MAC
//! `θ = |ε|/|r−c| ≤ 0.5` this is ~10⁻¹. Sufficient for smooth /
//! positive-dominated source patterns; bump to spherical-harmonic
//! order `P ≥ 6` if GMRES convergence degrades on a heterogeneous
//! mesh.

use crate::solver::panel_integrals::FOUR_PI;
use glam::DVec3;

/// Single-layer (monopole + dipole + quadrupole) multipole of a set
/// of point sources about a cluster centre. Per-cluster storage:
/// 1 + 3 + 6 = 10 f64s (the quadrupole tensor is symmetric, so we
/// store the upper-triangle once).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Expansion {
    pub(super) center: DVec3,
    pub(super) q: f64,
    pub(super) p: DVec3,
    // Upper-triangular quadrupole tensor components: xx, xy, xz, yy, yz, zz.
    pub(super) qxx: f64,
    pub(super) qxy: f64,
    pub(super) qxz: f64,
    pub(super) qyy: f64,
    pub(super) qyz: f64,
    pub(super) qzz: f64,
}

impl Expansion {
    /// Build from `N` sources `(s_i, q_i)` around `center`. Purely
    /// additive over sources — safe to fold in parallel via
    /// [`Self::fold`] if the caller already has per-worker partials.
    /// Test-only helper; production builds moments via the tree's
    /// upward sweep (see `treecode::build_expansions`).
    #[cfg(test)]
    pub(super) fn from_sources(center: DVec3, sources: &[(DVec3, f64)]) -> Self {
        let mut exp = Self {
            center,
            ..Self::default()
        };
        for &(s, q) in sources {
            exp.accumulate(s, q);
        }
        exp
    }

    /// Accumulate one source.
    pub(super) fn accumulate(&mut self, s: DVec3, q: f64) {
        let d = s - self.center;
        self.q += q;
        self.p += q * d;
        self.qxx += q * d.x * d.x;
        self.qxy += q * d.x * d.y;
        self.qxz += q * d.x * d.z;
        self.qyy += q * d.y * d.y;
        self.qyz += q * d.y * d.z;
        self.qzz += q * d.z * d.z;
    }

    /// Evaluate `Σ q_i / (4π |r − s_i|)` at target `r` using the
    /// Taylor expansion around `self.center`. Requires
    /// `|r − center| ≥ cluster_bounding_radius` for the truncated
    /// series to converge.
    pub(super) fn evaluate_laplace(&self, r: DVec3) -> f64 {
        let d = r - self.center;
        let r2 = d.length_squared();
        let inv_r = r2.sqrt().recip();
        let inv_r3 = inv_r * inv_r * inv_r;
        let inv_r5 = inv_r3 * inv_r * inv_r;

        let monopole = self.q * inv_r;
        let dipole = self.p.dot(d) * inv_r3;

        // `q_dd` is the full symmetric contraction `Σ_ij Q_ij d_i d_j`;
        // off-diagonal entries enter twice because the tensor is
        // symmetric. `tr_q = Q_xx + Q_yy + Q_zz` folds the δ_ij trace.
        let q_dd = symmetric_contract(self, d);
        let tr_q = self.qxx + self.qyy + self.qzz;
        // H_ij^0 = (3 d_i d_j − r² δ_ij) / r⁵
        let quadrupole = 0.5 * (3.0 * q_dd - r2 * tr_q) * inv_r5;

        (monopole + dipole + quadrupole) / FOUR_PI
    }

    /// Evaluate `Σ q_i · exp(−κ|r − s_i|) / (4π |r − s_i|)` at target
    /// `r`. Same convergence condition as [`Self::evaluate_laplace`];
    /// reduces to it at κ = 0 (within fp roundoff).
    pub(super) fn evaluate_yukawa(&self, r: DVec3, kappa: f64) -> f64 {
        let d = r - self.center;
        let r2 = d.length_squared();
        let r_mag = r2.sqrt();
        let inv_r = r_mag.recip();
        let inv_r3 = inv_r * inv_r * inv_r;
        let inv_r5 = inv_r3 * inv_r * inv_r;

        let kr = kappa * r_mag;
        let exp_kr = (-kr).exp();
        // Kernel derivatives for Yukawa evaluated at `d`:
        //   ∂G_κ/∂s_i    = (1+κR)·exp(−κR)·d_i / (4πR³)              =:  B / (4πR³) · d_i
        //   ∂²G_κ/∂s_i∂s_j = exp(−κR)/(4πR⁵) · [A d_i d_j − B R² δ_ij]
        // with B = 1 + κR, A = 3 + 3κR + (κR)². At κ=0, B=1 and A=3,
        // recovering the Laplace Hessian `3 d_i d_j − R² δ_ij`.
        let b_coef = 1.0 + kr;
        let a_coef = b_coef.mul_add(3.0, kr * kr);

        let monopole = self.q * exp_kr * inv_r;
        let dipole = self.p.dot(d) * b_coef * exp_kr * inv_r3;
        let q_dd = symmetric_contract(self, d);
        let tr_q = self.qxx + self.qyy + self.qzz;
        let quadrupole = 0.5 * exp_kr * (a_coef * q_dd - b_coef * r2 * tr_q) * inv_r5;

        (monopole + dipole + quadrupole) / FOUR_PI
    }

    /// Merge another expansion (same `center`) into `self`. Used by
    /// the M2M upward sweep when a parent cluster gathers its
    /// children after they've been translated to the parent centre.
    pub(super) fn fold(&mut self, other: &Self) {
        debug_assert!(
            (self.center - other.center).length_squared() < 1e-24,
            "fold only valid at identical centres"
        );
        self.q += other.q;
        self.p += other.p;
        self.qxx += other.qxx;
        self.qxy += other.qxy;
        self.qxz += other.qxz;
        self.qyy += other.qyy;
        self.qyz += other.qyz;
        self.qzz += other.qzz;
    }

    /// Translate the expansion to a new centre. Applies the closed-form
    /// shift formulas for a Taylor-order-2 multipole:
    ///
    /// ```text
    /// Q'       = Q
    /// p'_i     = p_i − Q · δ_i
    /// Q'_{ij}  = Q_{ij} − δ_j p_i − δ_i p_j + δ_i δ_j Q
    /// ```
    ///
    /// where `δ = new_center − self.center` is the shift. This is the
    /// M2M translation for an order-2 cartesian expansion; derived by
    /// rewriting `Σ q_α (s_α − c_new)^{⊗k}` using
    /// `s_α − c_new = (s_α − c_old) − δ`.
    pub(super) fn shifted_to(self, new_center: DVec3) -> Self {
        let d = new_center - self.center;
        let q = self.q;
        let p = self.p - q * d;
        // why: symmetric tensor shift expanded component-wise; pairing
        // `δ_j p_i + δ_i p_j` into the single off-diagonal entry keeps
        // the six-component upper-triangle storage consistent.
        let qxx = self.qxx - 2.0 * d.x * self.p.x + d.x * d.x * q;
        let qyy = self.qyy - 2.0 * d.y * self.p.y + d.y * d.y * q;
        let qzz = self.qzz - 2.0 * d.z * self.p.z + d.z * d.z * q;
        let qxy = self.qxy - d.x * self.p.y - d.y * self.p.x + d.x * d.y * q;
        let qxz = self.qxz - d.x * self.p.z - d.z * self.p.x + d.x * d.z * q;
        let qyz = self.qyz - d.y * self.p.z - d.z * self.p.y + d.y * d.z * q;
        Self {
            center: new_center,
            q,
            p,
            qxx,
            qxy,
            qxz,
            qyy,
            qyz,
            qzz,
        }
    }
}

/// Full symmetric contraction `Σ_{i,j} Q_{ij} d_i d_j` with the
/// off-diagonal components entering twice (tensor is symmetric).
/// Free function so both evaluators share identical inlined code.
#[inline]
fn symmetric_contract(e: &Expansion, d: DVec3) -> f64 {
    e.qxx * d.x * d.x
        + e.qyy * d.y * d.y
        + e.qzz * d.z * d.z
        + 2.0 * (e.qxy * d.x * d.y + e.qxz * d.x * d.z + e.qyz * d.y * d.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_sum(sources: &[(DVec3, f64)], target: DVec3) -> f64 {
        sources
            .iter()
            .map(|&(s, q)| q / (FOUR_PI * (target - s).length()))
            .sum()
    }

    #[test]
    fn single_source_at_center_is_exact() {
        let c = DVec3::new(1.0, 2.0, 3.0);
        let sources = vec![(c, 5.0)];
        let exp = Expansion::from_sources(c, &sources);
        let target = DVec3::new(10.0, 10.0, 10.0);
        let got = exp.evaluate_laplace(target);
        let expected = direct_sum(&sources, target);
        assert!(
            (got - expected).abs() < 1e-14,
            "monopole at centre should be exact: got={got}, expected={expected}"
        );
    }

    #[test]
    fn yukawa_at_kappa_zero_matches_laplace() {
        // With `κ = 0` the Yukawa evaluator reduces mathematically
        // to the Laplace one (exp(0) = 1, B = 1, A = 3). Enforce
        // bit-close agreement — the two paths share the same
        // moment-contraction code, so the only source of drift is
        // the multiplication by `exp(0)`.
        let center = DVec3::new(0.1, 0.2, -0.3);
        let sources: Vec<(DVec3, f64)> = (0..6)
            .map(|i| (center + DVec3::splat(0.05 * i as f64), 1.0 + 0.2 * i as f64))
            .collect();
        let exp = Expansion::from_sources(center, &sources);
        let target = DVec3::new(5.0, -3.0, 4.0);
        let laplace = exp.evaluate_laplace(target);
        let yukawa = exp.evaluate_yukawa(target, 0.0);
        assert!(
            (laplace - yukawa).abs() < 1e-15,
            "κ=0 disagreement: laplace={laplace}, yukawa={yukawa}"
        );
    }

    #[test]
    fn yukawa_single_source_at_center_is_exact() {
        // A single source at the expansion centre collapses the
        // dipole and quadrupole to zero; the monopole term alone
        // reproduces `exp(−κR)/(4πR)` exactly.
        let c = DVec3::new(1.0, 2.0, 3.0);
        let kappa = 0.125;
        let exp = Expansion::from_sources(c, &[(c, 5.0)]);
        let target = DVec3::new(10.0, 10.0, 10.0);
        let r = (target - c).length();
        let expected = 5.0 * (-kappa * r).exp() / (FOUR_PI * r);
        let got = exp.evaluate_yukawa(target, kappa);
        assert!(
            (got - expected).abs() < 1e-14,
            "single source: got={got}, expected={expected}"
        );
    }

    #[test]
    fn far_field_error_decays_monotonically() {
        // Cluster of positive charges in a unit sphere; evaluate at
        // growing distances. The *relative* error can decay at
        // different rates depending on how much monopole / dipole
        // cancellation the distribution has, so the invariant we
        // check is just: error strictly decreases, and becomes small
        // once R ≥ 8·cluster_radius.
        let sources: Vec<(DVec3, f64)> = (0..16)
            .map(|i| {
                let t = i as f64 * 0.137;
                (
                    DVec3::new(t.sin(), t.cos() * 1.7, (t * 2.0).cos()) * 0.5,
                    1.0 + 0.1 * (i as f64),
                )
            })
            .collect();
        let exp = Expansion::from_sources(DVec3::ZERO, &sources);

        let mut prev = f64::INFINITY;
        for distance in [5.0_f64, 10.0, 20.0, 40.0] {
            let target = DVec3::new(distance, 0.3 * distance, -0.1 * distance);
            let approx = exp.evaluate_laplace(target);
            let exact = direct_sum(&sources, target);
            let rel = (approx - exact).abs() / exact.abs();
            assert!(
                rel < prev,
                "error should decrease with distance; at d={distance} rel={rel:e}, prev={prev:e}"
            );
            prev = rel;
        }
        // With an all-positive cluster, the monopole dominates and
        // order-2 truncation gets ~6 digits at R = 40·ε.
        assert!(prev < 1e-5, "far-field error too large: {prev:e}");
    }

    #[test]
    fn shift_matches_rebuild_from_sources() {
        // The shift formulas must yield moments bit-identical (up to
        // fp associativity) to rebuilding the expansion at the new
        // centre from the original sources. This is the right
        // invariant — each Taylor-order-2 expansion is still an
        // approximation; two expansions at different centres will
        // differ at higher order when evaluated.
        let c1 = DVec3::new(0.1, 0.2, 0.3);
        let c2 = DVec3::new(1.0, -0.5, 0.8);
        let sources: Vec<(DVec3, f64)> = (0..8)
            .map(|i| {
                let t = i as f64 * 0.3;
                (
                    c1 + DVec3::new(t.sin(), t.cos(), (2.0 * t).sin()) * 0.1,
                    1.0,
                )
            })
            .collect();
        let shifted = Expansion::from_sources(c1, &sources).shifted_to(c2);
        let rebuilt = Expansion::from_sources(c2, &sources);
        for (got, want) in [
            (shifted.q, rebuilt.q),
            (shifted.p.x, rebuilt.p.x),
            (shifted.p.y, rebuilt.p.y),
            (shifted.p.z, rebuilt.p.z),
            (shifted.qxx, rebuilt.qxx),
            (shifted.qxy, rebuilt.qxy),
            (shifted.qxz, rebuilt.qxz),
            (shifted.qyy, rebuilt.qyy),
            (shifted.qyz, rebuilt.qyz),
            (shifted.qzz, rebuilt.qzz),
        ] {
            assert!(
                (got - want).abs() < 1e-12 * (1.0 + want.abs()),
                "shift/rebuild mismatch: got={got}, want={want}"
            );
        }
    }

    #[test]
    fn fold_equivalent_to_single_accumulate() {
        let c = DVec3::new(0.5, -0.2, 1.0);
        let sources: Vec<(DVec3, f64)> = (0..10)
            .map(|i| (c + DVec3::splat(0.1 * i as f64), 1.0 + 0.3 * i as f64))
            .collect();
        let monolith = Expansion::from_sources(c, &sources);

        let (a, b) = sources.split_at(sources.len() / 2);
        let ea = Expansion::from_sources(c, a);
        let eb = Expansion::from_sources(c, b);
        let mut merged = ea;
        merged.fold(&eb);

        // Fields should be bit-identical up to fp non-associativity.
        for (got, want) in [
            (merged.q, monolith.q),
            (merged.p.x, monolith.p.x),
            (merged.p.y, monolith.p.y),
            (merged.p.z, monolith.p.z),
            (merged.qxx, monolith.qxx),
            (merged.qxy, monolith.qxy),
            (merged.qxz, monolith.qxz),
            (merged.qyy, monolith.qyy),
            (merged.qyz, monolith.qyz),
            (merged.qzz, monolith.qzz),
        ] {
            assert!(
                (got - want).abs() < 1e-14,
                "fold mismatch: got={got}, want={want}"
            );
        }
    }
}
