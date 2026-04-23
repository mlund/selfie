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

use crate::solver::panel_integrals::{FOUR_PI, exp_neg};
use glam::DVec3;

/// Multipole moments about a cluster centre, carrying both the
/// single-layer moments (for `K·h` evaluation) and the double-layer
/// moments (for `K'·f` evaluation). Per-cluster storage:
/// - Single-layer: `1 + 3 + 6 = 10` f64 (monopole `Q`, dipole `p`,
///   symmetric quadrupole `Q_ij`).
/// - Double-layer order 1: `3 + 6 = 9` f64 (dipole `D_α`, symmetric
///   mixed tensor `M_αβ`). The raw `Σ q n_α (s−c)_β` tensor is
///   non-symmetric, but it is always contracted with the symmetric
///   Hessian `∂²G/∂s_α∂s_β`, so only its symmetric part matters.
///   We symmetrise at accumulate time to keep storage tight and
///   matching the quadrupole's upper-triangle layout.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Expansion {
    pub(super) center: DVec3,
    // -- Single-layer moments (Σ q (s−c)^⊗k) --
    pub(super) q: f64,
    pub(super) p: DVec3,
    // Upper-triangular quadrupole tensor components: xx, xy, xz, yy, yz, zz.
    pub(super) qxx: f64,
    pub(super) qxy: f64,
    pub(super) qxz: f64,
    pub(super) qyy: f64,
    pub(super) qyz: f64,
    pub(super) qzz: f64,
    // -- Double-layer moments (Σ q n (s−c)^⊗k, symmetrised in α↔β) --
    /// Dipole `D_α = Σ q_b · n_{b,α}` (centre-independent).
    pub(super) d: DVec3,
    // Symmetric part of `M_αβ = Σ q_b · n_{b,α} · (s_b − c)_β`:
    //   M_sym_αβ = ½·Σ q_b · (n_α (s−c)_β + n_β (s−c)_α)
    // Upper-triangle storage: xx, xy, xz, yy, yz, zz.
    pub(super) mxx: f64,
    pub(super) mxy: f64,
    pub(super) mxz: f64,
    pub(super) myy: f64,
    pub(super) myz: f64,
    pub(super) mzz: f64,
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

    /// Accumulate one single-layer source.
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

    /// Accumulate one double-layer source: a panel with density `q`,
    /// outward unit normal `n`, at position `s`. The symmetric `M`
    /// tensor stores `½(n_α (s−c)_β + n_β (s−c)_α)·q`.
    pub(super) fn accumulate_dipole(&mut self, s: DVec3, q: f64, n: DVec3) {
        let d = s - self.center;
        self.d += q * n;
        self.mxx += q * n.x * d.x;
        self.myy += q * n.y * d.y;
        self.mzz += q * n.z * d.z;
        self.mxy += 0.5 * q * (n.x * d.y + n.y * d.x);
        self.mxz += 0.5 * q * (n.x * d.z + n.z * d.x);
        self.myz += 0.5 * q * (n.y * d.z + n.z * d.y);
    }

    /// Evaluate `Σ q_i / (4π |r − s_i|)` at target `r` using the
    /// Taylor expansion around `self.center`. Requires
    /// `|r − center| ≥ cluster_bounding_radius` for the truncated
    /// series to converge.
    pub(super) fn evaluate_laplace(&self, r: DVec3) -> f64 {
        self.evaluate_laplace_pair(r).0
    }

    /// Double-layer Laplace: evaluate
    /// `Σ_b q_b · n_b · ∇_s G₀(r, s_b)` at target `r`. Uses the
    /// `(D, M)` moments only.
    pub(super) fn evaluate_double_laplace(&self, r: DVec3) -> f64 {
        self.evaluate_laplace_pair(r).1
    }

    /// Double-layer Yukawa version of
    /// [`Self::evaluate_double_laplace`].
    pub(super) fn evaluate_double_yukawa(&self, r: DVec3, kappa: f64) -> f64 {
        self.evaluate_yukawa_pair(r, kappa).1
    }

    /// Combined Laplace evaluator: returns `(single, double)` at
    /// target `r` in one pass, sharing the distance-power chain
    /// `d, r², 1/r, 1/r³, 1/r⁵`. Production callers (the tree
    /// traversal) use this; the per-kernel entry points above are
    /// convenience shims.
    pub(super) fn evaluate_laplace_pair(&self, r: DVec3) -> (f64, f64) {
        let d = r - self.center;
        let r2 = d.length_squared();
        let inv_r = r2.sqrt().recip();
        let inv_r3 = inv_r * inv_r * inv_r;
        let inv_r5 = inv_r3 * inv_r * inv_r;

        let p_dot_d = self.p.dot(d);
        let q_dd = symmetric_contract(
            self.qxx, self.qxy, self.qxz, self.qyy, self.qyz, self.qzz, d,
        );
        let q_tr = self.qxx + self.qyy + self.qzz;

        let d_dot_d = self.d.dot(d);
        let m_dd = symmetric_contract(
            self.mxx, self.mxy, self.mxz, self.myy, self.myz, self.mzz, d,
        );
        let m_tr = self.mxx + self.myy + self.mzz;

        // Single-layer: Q/r + p·d/r³ + ½(3 Q:dd − r² tr Q)/r⁵
        let single =
            (self.q * inv_r + p_dot_d * inv_r3 + 0.5 * (3.0 * q_dd - r2 * q_tr) * inv_r5) / FOUR_PI;
        // Double-layer: D·d/r³ + (3 M:dd − r² tr M)/r⁵
        let double = (d_dot_d * inv_r3 + (3.0 * m_dd - r2 * m_tr) * inv_r5) / FOUR_PI;
        (single, double)
    }

    /// Combined Yukawa evaluator: `(single, double)` sharing the
    /// distance-power chain AND the `exp(−κr)`, `A = 3+3κr+(κr)²`,
    /// `B = 1+κr` coefficients. One `sqrt` + one `exp` per call
    /// instead of two of each if both kernels were dispatched
    /// separately.
    pub(super) fn evaluate_yukawa_pair(&self, r: DVec3, kappa: f64) -> (f64, f64) {
        let d = r - self.center;
        let r2 = d.length_squared();
        let r_mag = r2.sqrt();
        let inv_r = r_mag.recip();
        let inv_r3 = inv_r * inv_r * inv_r;
        let inv_r5 = inv_r3 * inv_r * inv_r;

        let kr = kappa * r_mag;
        let exp_kr = exp_neg(kr);
        let b_coef = 1.0 + kr;
        let a_coef = b_coef.mul_add(3.0, kr * kr);

        let p_dot_d = self.p.dot(d);
        let q_dd = symmetric_contract(
            self.qxx, self.qxy, self.qxz, self.qyy, self.qyz, self.qzz, d,
        );
        let q_tr = self.qxx + self.qyy + self.qzz;

        let d_dot_d = self.d.dot(d);
        let m_dd = symmetric_contract(
            self.mxx, self.mxy, self.mxz, self.myy, self.myz, self.mzz, d,
        );
        let m_tr = self.mxx + self.myy + self.mzz;

        let single = exp_kr
            * (self.q * inv_r
                + b_coef * p_dot_d * inv_r3
                + 0.5 * (a_coef * q_dd - b_coef * r2 * q_tr) * inv_r5)
            / FOUR_PI;
        let double = exp_kr
            * (b_coef * d_dot_d * inv_r3 + (a_coef * m_dd - b_coef * r2 * m_tr) * inv_r5)
            / FOUR_PI;
        (single, double)
    }

    /// Evaluate `Σ q_i · exp(−κ|r − s_i|) / (4π |r − s_i|)` at target
    /// `r`. Same convergence condition as [`Self::evaluate_laplace`];
    /// reduces to it at κ = 0 (within fp roundoff).
    pub(super) fn evaluate_yukawa(&self, r: DVec3, kappa: f64) -> f64 {
        self.evaluate_yukawa_pair(r, kappa).0
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
        self.d += other.d;
        self.mxx += other.mxx;
        self.mxy += other.mxy;
        self.mxz += other.mxz;
        self.myy += other.myy;
        self.myz += other.myz;
        self.mzz += other.mzz;
    }

    /// Translate the expansion to a new centre. Applies the closed-form
    /// shift formulas for single-layer (order 2) and double-layer
    /// (order 1):
    ///
    /// ```text
    /// Single-layer:
    ///   Q'     = Q
    ///   p'_i   = p_i − Q · δ_i
    ///   Q'_ij  = Q_ij − δ_j p_i − δ_i p_j + δ_i δ_j Q
    ///
    /// Double-layer:
    ///   D'     = D                           (centre-independent)
    ///   M'_αβ  = M_αβ − δ_β D_α
    /// ```
    ///
    /// where `δ = new_center − self.center`. Derived by rewriting each
    /// moment in the new frame and expanding
    /// `s − c_new = (s − c_old) − δ`.
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
        // Double-layer: D is translation-invariant; the symmetric
        // `M_sym_αβ` gains the symmetric rank-1 correction
        //   − (δ_β D_α + δ_α D_β) / 2
        // derived from the underlying non-symmetric `M_αβ − δ_β D_α`
        // after symmetrising.
        let dipole = self.d;
        let mxx = self.mxx - d.x * dipole.x;
        let myy = self.myy - d.y * dipole.y;
        let mzz = self.mzz - d.z * dipole.z;
        let mxy = self.mxy - 0.5 * (d.y * dipole.x + d.x * dipole.y);
        let mxz = self.mxz - 0.5 * (d.z * dipole.x + d.x * dipole.z);
        let myz = self.myz - 0.5 * (d.z * dipole.y + d.y * dipole.z);
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
            d: dipole,
            mxx,
            mxy,
            mxz,
            myy,
            myz,
            mzz,
        }
    }
}

/// Full symmetric contraction `Σ_{i,j} T_{ij} d_i d_j` with the
/// off-diagonal components entering twice (tensor is symmetric).
/// Takes the 6 upper-triangle components explicitly so both the
/// single-layer `Q` and the double-layer `M_sym` can use it.
#[inline]
fn symmetric_contract(xx: f64, xy: f64, xz: f64, yy: f64, yz: f64, zz: f64, d: DVec3) -> f64 {
    xx.mul_add(
        d.x * d.x,
        yy.mul_add(
            d.y * d.y,
            zz.mul_add(
                d.z * d.z,
                2.0 * (xy * d.x * d.y + xz * d.x * d.z + yz * d.y * d.z),
            ),
        ),
    )
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

    fn direct_double_sum(sources: &[(DVec3, f64, DVec3)], target: DVec3, kappa: f64) -> f64 {
        // Σ q n · ∇_s G(r, s)  where G is Laplace (κ=0) or Yukawa.
        sources
            .iter()
            .map(|&(s, q, n)| {
                let d = target - s;
                let r = d.length();
                let r3 = r * r * r;
                let gfactor = if kappa == 0.0 {
                    1.0
                } else {
                    (1.0 + kappa * r) * (-kappa * r).exp()
                };
                q * n.dot(d) * gfactor / (FOUR_PI * r3)
            })
            .sum()
    }

    #[test]
    fn double_layer_at_kappa_zero_matches_laplace() {
        // evaluate_double_yukawa reduces to evaluate_double_laplace at
        // κ = 0 (by identical algebra — the only difference is the
        // exp(0) = 1 multiplication).
        let center = DVec3::new(0.1, 0.2, -0.3);
        let mut exp = Expansion {
            center,
            ..Expansion::default()
        };
        for i in 0..6 {
            let s = center + DVec3::splat(0.05 * i as f64);
            let n = DVec3::new(
                (i as f64 * 0.7).sin(),
                (i as f64 * 0.7).cos(),
                (i as f64 * 0.3).sin(),
            )
            .normalize();
            exp.accumulate_dipole(s, 1.0 + 0.2 * i as f64, n);
        }
        let target = DVec3::new(5.0, -3.0, 4.0);
        let laplace = exp.evaluate_double_laplace(target);
        let yukawa = exp.evaluate_double_yukawa(target, 0.0);
        assert!(
            (laplace - yukawa).abs() < 1e-15,
            "κ=0: laplace={laplace}, yukawa={yukawa}"
        );
    }

    #[test]
    fn double_layer_single_source_at_center_is_exact() {
        // With one source exactly at the centre, M collapses and only
        // the D·∇G term survives; the truncated expansion reproduces
        // the true dipole evaluation exactly.
        let c = DVec3::new(1.0, 2.0, 3.0);
        let n = DVec3::new(0.0, 0.0, 1.0);
        let mut exp = Expansion {
            center: c,
            ..Expansion::default()
        };
        exp.accumulate_dipole(c, 2.5, n);
        let target = DVec3::new(10.0, 10.0, 10.0);
        let got = exp.evaluate_double_laplace(target);
        let expected = direct_double_sum(&[(c, 2.5, n)], target, 0.0);
        assert!(
            (got - expected).abs() < 1e-14,
            "got={got}, expected={expected}"
        );
    }

    #[test]
    fn double_layer_far_field_decays_with_distance() {
        // A cluster of point-dipoles in a small ball; evaluate at
        // growing distances and verify truncation error shrinks
        // monotonically (the sanity check that matched the single-
        // layer test).
        let center = DVec3::ZERO;
        let mut exp = Expansion {
            center,
            ..Expansion::default()
        };
        let sources: Vec<(DVec3, f64, DVec3)> = (0..12)
            .map(|i| {
                let t = i as f64 * 0.25;
                let s = DVec3::new(t.sin(), t.cos() * 0.8, (2.0 * t).sin()) * 0.5;
                let n = DVec3::new(t.cos(), -(t.sin()), (t * 3.0).sin()).normalize();
                (s, 1.0 + 0.1 * i as f64, n)
            })
            .collect();
        for &(s, q, n) in &sources {
            exp.accumulate_dipole(s, q, n);
        }
        let mut prev = f64::INFINITY;
        for distance in [4.0_f64, 10.0, 25.0] {
            let target = DVec3::new(distance, 0.3 * distance, -0.1 * distance);
            let approx = exp.evaluate_double_laplace(target);
            let exact = direct_double_sum(&sources, target, 0.0);
            let rel = (approx - exact).abs() / exact.abs();
            assert!(
                rel < prev,
                "error should shrink: d={distance} rel={rel:e}, prev={prev:e}"
            );
            prev = rel;
        }
        // Order-1 double-layer at θ ≈ 0.02 should be well under 1 %.
        assert!(prev < 1e-3, "far-field error too large: {prev:e}");
    }

    #[test]
    fn double_layer_shift_matches_rebuild() {
        // Accumulate at one centre, shift to another, compare with a
        // re-accumulation from scratch at the new centre. The shift
        // formulas for D and M_sym must preserve the moments exactly.
        let c1 = DVec3::new(0.1, 0.2, 0.3);
        let c2 = DVec3::new(1.0, -0.5, 0.8);
        let sources: Vec<(DVec3, f64, DVec3)> = (0..6)
            .map(|i| {
                let t = i as f64 * 0.3;
                let s = c1 + DVec3::new(t.sin(), t.cos(), (2.0 * t).sin()) * 0.1;
                let n = DVec3::new(t.cos(), -t.sin(), (t + 0.5).sin()).normalize();
                (s, 1.0, n)
            })
            .collect();
        let mut at_c1 = Expansion {
            center: c1,
            ..Expansion::default()
        };
        let mut at_c2 = Expansion {
            center: c2,
            ..Expansion::default()
        };
        for &(s, q, n) in &sources {
            at_c1.accumulate_dipole(s, q, n);
            at_c2.accumulate_dipole(s, q, n);
        }
        let shifted = at_c1.shifted_to(c2);
        for (got, want, name) in [
            (shifted.d.x, at_c2.d.x, "D.x"),
            (shifted.d.y, at_c2.d.y, "D.y"),
            (shifted.d.z, at_c2.d.z, "D.z"),
            (shifted.mxx, at_c2.mxx, "mxx"),
            (shifted.myy, at_c2.myy, "myy"),
            (shifted.mzz, at_c2.mzz, "mzz"),
            (shifted.mxy, at_c2.mxy, "mxy"),
            (shifted.mxz, at_c2.mxz, "mxz"),
            (shifted.myz, at_c2.myz, "myz"),
        ] {
            assert!(
                (got - want).abs() < 1e-12 * (1.0 + want.abs()),
                "{name}: shifted={got}, rebuilt={want}"
            );
        }
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
        // Tolerance set by `exp_neg`'s 6th-order Taylor (~1e-7 relative),
        // not by the multipole math (which is exact when the sole
        // source sits at the centre).
        let rel = (got - expected).abs() / expected.abs();
        assert!(
            rel < 1e-6,
            "single source: got={got}, expected={expected}, rel={rel:e}"
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
