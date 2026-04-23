//! Complex solid spherical harmonics and a matching multipole
//! expansion for the Laplace single-layer kernel.
//!
//! Convention (Epton-Dembart / Greengard-Rokhlin compatible): for
//! `0 ≤ m ≤ n`,
//! ```text
//! R_n^m(r) = (−1)^m · r^n · P_n^m(cos θ) · e^{imφ} / (n + m)!
//! S_n^m(r) = (−1)^m · (n − m)! · P_n^m(cos θ) · e^{imφ} / r^(n+1)
//! ```
//! and `R_n^{−m} = (−1)^m · conj(R_n^m)`, `S_n^{−m} = (−1)^m · conj(S_n^m)`.
//! The `(n + m)!` scaling on `R` and `(n − m)!` on `S` keeps
//! coefficients in a numerically tame range and makes the addition
//! theorem coefficient-free:
//! ```text
//! 1 / |r − s| = Σ_{n=0..∞} Σ_{m=−n..n} conj(R_n^m(s)) · S_n^m(r)
//! ```
//! valid whenever `|s| < |r|`. Truncating at `n ≤ P` gives relative
//! error `~(|s|/|r|)^(P+1)`.
//!
//! Per-cluster storage is `(P + 1)² = 49` complex coefficients at
//! `P = 6`, ≈ 800 B per expansion — fits in L2 for any BEM mesh we
//! will realistically tree-partition.

use crate::solver::panel_integrals::FOUR_PI;
use faer::c64;
use glam::DVec3;

/// Spherical-harmonic expansion order.
pub(super) const P: usize = 6;
/// Table length for a full `(n, m)` pair with `0 ≤ n ≤ P, |m| ≤ n`.
pub(super) const NCOEF: usize = (P + 1) * (P + 1);

/// Flat index `(n, m) → n² + n + m`, keyed by the non-negative `m`
/// slot. The row for degree `n` spans `[n² .. n² + 2n + 1)`; `m = 0`
/// sits at the centre `n² + n`, and `±m` slots are equidistant from
/// it. Negative-m writes just subtract in place (see the mirror
/// block at the bottom of [`regular_solid_harmonic`]).
#[inline]
pub(super) const fn idx(n: usize, m: usize) -> usize {
    n * n + n + m
}

/// Full `(n, m)` table of regular solid harmonics `R_n^m(r)` for
/// `0 ≤ n ≤ P`, `|m| ≤ n`. Result indexing is [`idx`].
///
/// Computed via the pure-solid-harmonic recurrences
/// ```text
/// R_0^0 = 1
/// R_m^m       = (x + iy) · R_{m−1}^{m−1} / (2m)                          m ≥ 1
/// R_{m+1}^m   = z · R_m^m                                                (column seed)
/// (l+1−m)(l+1+m) · R_{l+1}^m = (2l+1) · z · R_l^m − r² · R_{l−1}^m       l ≥ m
/// ```
/// derived from the standard associated-Legendre 3-term recurrence;
/// the `(n + m)!` scaling absorbs Legendre's factorial growth so the
/// coefficients stay `O(r^n)`.
pub(super) fn regular_solid_harmonic(r: DVec3) -> [c64; NCOEF] {
    let mut out = [c64::new(0.0, 0.0); NCOEF];
    out[idx(0, 0)] = c64::new(1.0, 0.0);

    let xy = c64::new(r.x, r.y);
    let z = r.z;
    let r2 = r.length_squared();

    for m in 1..=P {
        let prev = out[idx(m - 1, m - 1)];
        out[idx(m, m)] = xy * prev * (0.5 / m as f64);
    }

    for m in 0..=P {
        if m < P {
            let prev = out[idx(m, m)];
            out[idx(m + 1, m)] = prev * z;
        }
        for n in (m + 2)..=P {
            let l = n - 1;
            let twolp1 = (2 * l + 1) as f64;
            let inv_denom = 1.0 / ((l - m + 1) * (l + m + 1)) as f64;
            let r_l = out[idx(l, m)];
            let r_lm1 = out[idx(l - 1, m)];
            out[idx(n, m)] = (r_l * (z * twolp1) - r_lm1 * r2) * inv_denom;
        }
    }

    mirror_negative_m(&mut out);
    out
}

/// Full `(n, m)` table of irregular solid harmonics `S_n^m(r)` for
/// `0 ≤ n ≤ P`, `|m| ≤ n`. Debug-asserts `|r| > 0` — `S_n^m` diverges
/// at the origin.
///
/// Recurrences:
/// ```text
/// S_0^0 = 1 / r
/// S_m^m     = (2m − 1) · (x + iy) · S_{m−1}^{m−1} / r²                   m ≥ 1
/// S_{m+1}^m = (2m + 1) · z · S_m^m / r²                                  (column seed)
/// r² · S_{l+1}^m = (2l+1) · z · S_l^m − (l² − m²) · S_{l−1}^m            l ≥ m
/// ```
pub(super) fn irregular_solid_harmonic(r: DVec3) -> [c64; NCOEF] {
    let r2 = r.length_squared();
    debug_assert!(r2 > 0.0, "irregular solid harmonic diverges at r = 0");
    let inv_r = r2.sqrt().recip();
    let inv_r2 = 1.0 / r2;

    let mut out = [c64::new(0.0, 0.0); NCOEF];
    out[idx(0, 0)] = c64::new(inv_r, 0.0);

    let xy = c64::new(r.x, r.y);
    let z = r.z;

    for m in 1..=P {
        let prev = out[idx(m - 1, m - 1)];
        out[idx(m, m)] = xy * prev * ((2 * m - 1) as f64 * inv_r2);
    }

    for m in 0..=P {
        if m < P {
            let prev = out[idx(m, m)];
            out[idx(m + 1, m)] = prev * (z * (2 * m + 1) as f64 * inv_r2);
        }
        for n in (m + 2)..=P {
            let l = n - 1;
            let twolp1 = (2 * l + 1) as f64;
            let ll_mm = (l * l - m * m) as f64;
            let s_l = out[idx(l, m)];
            let s_lm1 = out[idx(l - 1, m)];
            out[idx(n, m)] = (s_l * (z * twolp1) - s_lm1 * ll_mm) * inv_r2;
        }
    }

    mirror_negative_m(&mut out);
    out
}

/// Fill the negative-`m` slots from the already-populated positive-`m`
/// ones via `R_n^{−m} = (−1)^m · conj(R_n^m)` (identical identity for
/// S). Writing in place: `m = 0` sits at row-centre `n² + n`; the `+m`
/// slot is at `+m` offset, the `−m` slot at `−m`.
fn mirror_negative_m(out: &mut [c64; NCOEF]) {
    for n in 1..=P {
        let row = n * n + n;
        for m in 1..=n {
            let pos = out[row + m];
            let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
            out[row - m] = pos.conj() * sign;
        }
    }
}

/// Spherical-harmonic multipole expansion, single-layer Laplace only
/// at this stage. Moments are
/// ```text
/// M_n^m = Σ_i q_i · conj(R_n^m(s_i − center))
/// ```
/// and [`Self::evaluate_laplace`] contracts them against the
/// irregular harmonic at the target, dividing by `4π` to match the
/// solver's `G_0(r, s) = 1 / (4π |r − s|)` normalization.
#[derive(Clone, Copy, Debug)]
pub(super) struct ShExpansion {
    pub(super) center: DVec3,
    pub(super) coeffs: [c64; NCOEF],
}

impl Default for ShExpansion {
    fn default() -> Self {
        Self {
            center: DVec3::ZERO,
            coeffs: [c64::new(0.0, 0.0); NCOEF],
        }
    }
}

impl ShExpansion {
    /// Accumulate one single-layer source `(s, q)` into the moments.
    pub(super) fn accumulate(&mut self, s: DVec3, q: f64) {
        let r = regular_solid_harmonic(s - self.center);
        for (coef, &rn) in self.coeffs.iter_mut().zip(r.iter()) {
            *coef += rn.conj() * q;
        }
    }

    /// Evaluate `Σ_i q_i / (4π |r − s_i|)` at target `r`. Caller must
    /// guarantee `|r − center| > max_i |s_i − center|` for the
    /// truncated series to converge — debug-asserted only via the
    /// underlying `r² > 0` check inside [`irregular_solid_harmonic`].
    pub(super) fn evaluate_laplace(&self, r: DVec3) -> f64 {
        let s = irregular_solid_harmonic(r - self.center);
        let mut sum = c64::new(0.0, 0.0);
        for (&coef, &sn) in self.coeffs.iter().zip(s.iter()) {
            sum += coef * sn;
        }
        // why: real-valued sources give M_n^{−m} = (−1)^m · conj(M_n^m);
        // paired with S_n^{−m} = (−1)^m · conj(S_n^m), cross-m terms
        // collapse to complex-conjugate pairs that cancel in the
        // imaginary part up to fp roundoff.
        debug_assert!(
            sum.im.abs() < 1e-6 * sum.re.abs().max(1.0),
            "Laplace SH evaluator: unexpected imaginary residual {:e}",
            sum.im
        );
        sum.re / FOUR_PI
    }

    /// Merge `other` into `self`, requiring identical centres. Used by
    /// the tree's upward sweep when a parent gathers child moments
    /// already translated to the parent centre via [`Self::shifted_to`].
    pub(super) fn fold(&mut self, other: &Self) {
        debug_assert!(
            (self.center - other.center).length_squared() < 1e-24,
            "fold only valid at identical centres"
        );
        for (a, b) in self.coeffs.iter_mut().zip(&other.coeffs) {
            *a += b;
        }
    }

    /// Translate the expansion to a new centre via the M2M operator:
    /// ```text
    /// M'_n^m(c₂) = Σ_{k=0..n} Σ_{j=−k..k} conj(R_k^j(−δ)) · M_{n−k}^{m−j}(c₁)
    /// ```
    /// with `δ = c₂ − c₁`, derived from the regular-harmonic addition
    /// theorem `R_n^m(s − δ) = Σ R_{n−k}^{m−j}(s) · R_k^j(−δ)`.
    /// `O(P⁴)` complex multiplies per shift.
    pub(super) fn shifted_to(&self, new_center: DVec3) -> Self {
        let delta = new_center - self.center;
        // why: conjugate the shift table up front (49 negations) so the
        // quadruple inner loop sees a bare complex multiply. Also lets
        // us fold the `.conj()` out of the hottest path.
        let r_conj: [c64; NCOEF] = regular_solid_harmonic(-delta).map(|c| c.conj());
        let mut new_coeffs = [c64::new(0.0, 0.0); NCOEF];
        for n in 0..=P {
            let n_sig = n as isize;
            let row_out = n * n + n;
            for m in -n_sig..=n_sig {
                let mut sum = c64::new(0.0, 0.0);
                for k in 0..=n {
                    let k_sig = k as isize;
                    let np = n - k;
                    let np_sig = np as isize;
                    // why: the underlying M_{np}^{mp} slot exists only
                    // for |mp| ≤ np. Clamping j's range up front avoids
                    // a conditional `continue` in the innermost loop.
                    let j_lo = (m - np_sig).max(-k_sig);
                    let j_hi = (m + np_sig).min(k_sig);
                    let row_k = k * k + k;
                    let row_np = np * np + np;
                    for j in j_lo..=j_hi {
                        let rc = r_conj[row_k.wrapping_add_signed(j)];
                        let mc = self.coeffs[row_np.wrapping_add_signed(m - j)];
                        sum += rc * mc;
                    }
                }
                new_coeffs[row_out.wrapping_add_signed(m)] = sum;
            }
        }
        Self {
            center: new_center,
            coeffs: new_coeffs,
        }
    }

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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-rolled direct sum of `q_i / (4π |r − s_i|)`.
    fn direct_sum(sources: &[(DVec3, f64)], target: DVec3) -> f64 {
        sources
            .iter()
            .map(|&(s, q)| q / (FOUR_PI * (target - s).length()))
            .sum()
    }

    #[test]
    fn monopole_anchors() {
        let probe = DVec3::new(1.5, -0.7, 2.3);
        let r = regular_solid_harmonic(probe);
        let s = irregular_solid_harmonic(probe);
        assert!((r[idx(0, 0)].re - 1.0).abs() < 1e-14);
        assert!(r[idx(0, 0)].im.abs() < 1e-14);
        let expected_s00 = 1.0 / probe.length();
        assert!((s[idx(0, 0)].re - expected_s00).abs() < 1e-14);
        assert!(s[idx(0, 0)].im.abs() < 1e-14);
    }

    #[test]
    fn r_1_closed_forms() {
        // R_1^0 = z, R_1^1 = (x + iy) / 2, R_1^{−1} = −(x − iy) / 2
        // from R_n^{−m} = (−1)^m · conj(R_n^m).
        let p = DVec3::new(1.3, -0.6, 2.1);
        let r = regular_solid_harmonic(p);
        // n=1 row centre = n² + n = 2; ±m slots are equidistant.
        let row1 = 2;
        assert!((r[row1].re - p.z).abs() < 1e-14);
        assert!(r[row1].im.abs() < 1e-14);
        assert!((r[row1 + 1].re - 0.5 * p.x).abs() < 1e-14);
        assert!((r[row1 + 1].im - 0.5 * p.y).abs() < 1e-14);
        assert!((r[row1 - 1].re + 0.5 * p.x).abs() < 1e-14);
        assert!((r[row1 - 1].im - 0.5 * p.y).abs() < 1e-14);
    }

    #[test]
    fn r_2_0_closed_form() {
        let p = DVec3::new(0.7, 1.1, -1.3);
        let r2 = p.length_squared();
        let r = regular_solid_harmonic(p);
        let expected = (3.0 * p.z * p.z - r2) * 0.25;
        assert!((r[idx(2, 0)].re - expected).abs() < 1e-14);
        assert!(r[idx(2, 0)].im.abs() < 1e-14);
    }

    #[test]
    fn addition_theorem_matches_direct_inverse_distance() {
        // Σ_{n,m} conj(R_n^m(s)) · S_n^m(r) must reproduce 1/|r − s|
        // when |s| < |r|, truncated at order P with error ~(|s|/|r|)^(P+1).
        let s = DVec3::new(0.3, -0.4, 0.5);
        let r = DVec3::new(4.0, 2.5, -3.0);
        let reg = regular_solid_harmonic(s);
        let irr = irregular_solid_harmonic(r);
        let mut sum = c64::new(0.0, 0.0);
        for (&rs, &ss) in reg.iter().zip(irr.iter()) {
            sum += rs.conj() * ss;
        }
        let expected = 1.0 / (r - s).length();
        let ratio = s.length() / r.length();
        let tol = 10.0 * ratio.powi(P as i32 + 1);
        assert!(
            sum.im.abs() < 1e-12,
            "addition-theorem imaginary residual: {:e}",
            sum.im
        );
        assert!(
            (sum.re - expected).abs() < tol,
            "got={}, expected={}, tol={:e}",
            sum.re,
            expected,
            tol
        );
    }

    #[test]
    fn sh_single_source_at_center_is_exact() {
        // Source at the centre: only M_0^0 is nonzero (R_n^m(0) = 0 for
        // n ≥ 1); the truncated evaluator reproduces 1/(4π|r−c|) exactly.
        let c = DVec3::new(1.0, 2.0, 3.0);
        let sources = vec![(c, 5.0)];
        let exp = ShExpansion::from_sources(c, &sources);
        let target = DVec3::new(10.0, 10.0, 10.0);
        let got = exp.evaluate_laplace(target);
        let expected = direct_sum(&sources, target);
        assert!(
            (got - expected).abs() < 1e-14,
            "monopole at centre: got={got}, expected={expected}"
        );
    }

    #[test]
    fn sh_far_field_error_decays_monotonically() {
        // 12 sources in a small ball about the origin; evaluate at a
        // growing sequence of distances and verify the relative error
        // shrinks. At P = 6 the Epton-Dembart truncation bound is
        // ~(ε/R)^7, so going from R = 5·ε to R = 40·ε should buy 7
        // decades of accuracy.
        let sources: Vec<(DVec3, f64)> = (0..12)
            .map(|i| {
                let t = i as f64 * 0.173;
                (
                    DVec3::new(t.sin(), t.cos() * 0.9, (t * 2.0).sin()) * 0.5,
                    1.0 + 0.15 * i as f64,
                )
            })
            .collect();
        let exp = ShExpansion::from_sources(DVec3::ZERO, &sources);

        let mut prev = f64::INFINITY;
        for distance in [5.0_f64, 10.0, 20.0, 40.0] {
            let target = DVec3::new(distance, 0.3 * distance, -0.1 * distance);
            let approx = exp.evaluate_laplace(target);
            let exact = direct_sum(&sources, target);
            let rel = (approx - exact).abs() / exact.abs();
            assert!(
                rel < prev,
                "d={distance}: rel={rel:e} not less than prev={prev:e}"
            );
            prev = rel;
        }
        // P = 6 should hit ~10⁻¹⁰ at R = 40·ε_cluster_radius (ε ≈ 0.5
        // gives (ε/R)^7 ≈ (1/80)^7 ≈ 1e-13, but fp roundoff in the
        // 49-term sum floors us around 1e-12).
        assert!(prev < 1e-10, "far-field error too large: {prev:e}");
    }

    #[test]
    fn sh_shift_matches_rebuild_from_sources() {
        // Accumulate at c₁, shift to c₂; compare coefficient-by-
        // coefficient against an expansion built fresh at c₂. The
        // M2M operator must preserve moments bit-for-bit up to fp
        // non-associativity.
        let c1 = DVec3::new(0.1, 0.2, 0.3);
        let c2 = DVec3::new(1.0, -0.5, 0.8);
        let sources: Vec<(DVec3, f64)> = (0..10)
            .map(|i| {
                let t = i as f64 * 0.237;
                (
                    c1 + DVec3::new(t.sin(), t.cos(), (2.0 * t).sin()) * 0.2,
                    1.0 + 0.1 * i as f64,
                )
            })
            .collect();
        let shifted = ShExpansion::from_sources(c1, &sources).shifted_to(c2);
        let rebuilt = ShExpansion::from_sources(c2, &sources);
        for (i, (&got, &want)) in shifted.coeffs.iter().zip(&rebuilt.coeffs).enumerate() {
            let diff = (got - want).norm();
            let scale = 1.0 + want.norm();
            assert!(
                diff < 1e-11 * scale,
                "coeff {i}: shifted={got:?}, rebuilt={want:?}, diff={diff:e}"
            );
        }
        assert!((shifted.center - c2).length() < 1e-14);
    }

    #[test]
    fn sh_fold_matches_single_accumulate() {
        let c = DVec3::new(0.5, -0.2, 1.0);
        let sources: Vec<(DVec3, f64)> = (0..10)
            .map(|i| (c + DVec3::splat(0.1 * i as f64), 1.0 + 0.3 * i as f64))
            .collect();
        let monolith = ShExpansion::from_sources(c, &sources);
        let (a_src, b_src) = sources.split_at(sources.len() / 2);
        let ea = ShExpansion::from_sources(c, a_src);
        let eb = ShExpansion::from_sources(c, b_src);
        let mut merged = ea;
        merged.fold(&eb);
        for (&got, &want) in merged.coeffs.iter().zip(&monolith.coeffs) {
            assert!((got - want).norm() < 1e-13);
        }
    }

    #[test]
    fn sh_matches_direct_on_sphere_distribution() {
        // Concrete gbe-adjacent scenario: 20 positive charges inside a
        // 1 Å ball, target at 8 Å along a generic axis. Checks the
        // full (n, m) mixing — all 49 coefficients carry weight.
        let sources: Vec<(DVec3, f64)> = (0..20)
            .map(|i| {
                let a = (i as f64) * 0.3;
                let b = (i as f64) * 0.7;
                (
                    DVec3::new(a.sin() * b.cos(), a.sin() * b.sin(), a.cos()) * 0.9,
                    0.5 + 0.1 * i as f64,
                )
            })
            .collect();
        let exp = ShExpansion::from_sources(DVec3::ZERO, &sources);
        let target = DVec3::new(4.5, -6.0, 3.2);
        let got = exp.evaluate_laplace(target);
        let exact = direct_sum(&sources, target);
        let rel = (got - exact).abs() / exact.abs();
        // ε/R ≈ 0.9/8 ≈ 0.11, so (ε/R)^7 ≈ 2e-7. Use a generous
        // factor of 10× for the multi-term aggregate.
        assert!(rel < 2e-6, "got={got}, exact={exact}, rel={rel:e}");
    }
}
