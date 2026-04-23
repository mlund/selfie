//! Complex solid spherical harmonics and a matching multipole
//! expansion for the Laplace single- and double-layer kernels.
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
    let mut out = [c64::default(); NCOEF];
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

    let mut out = [c64::default(); NCOEF];
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

/// Spherical-harmonic multipole expansion covering both the Laplace
/// single-layer and the Laplace double-layer kernels.
///
/// Single-layer moments:
/// ```text
/// M_n^m = Σ_i q_i · conj(R_n^m(s_i − center))
/// ```
/// Double-layer moments (source-side normal derivative):
/// ```text
/// Md_n^m = Σ_b f_b · (n_b · conj(∇_s R_n^m(s_b − center)))
/// ```
/// where `n_b · ∇_s R_n^m` expands via the closed-form recurrences
/// ```text
/// ∂R_n^m/∂x = ½ (R_{n−1}^{m−1} − R_{n−1}^{m+1})
/// ∂R_n^m/∂y = (i/2) (R_{n−1}^{m−1} + R_{n−1}^{m+1})
/// ∂R_n^m/∂z = R_{n−1}^m
/// ```
/// Both moment sets contract against the same `S_n^m(r − center)`
/// table at evaluate time; [`Self::evaluate_laplace_pair`] returns
/// `(single, double)` in one pass to share that table and the
/// `r − center` subtraction.
#[derive(Clone, Copy, Debug)]
pub(super) struct ShExpansion {
    pub(super) center: DVec3,
    pub(super) coeffs: [c64; NCOEF],
    pub(super) double_coeffs: [c64; NCOEF],
}

impl Default for ShExpansion {
    fn default() -> Self {
        Self {
            center: DVec3::ZERO,
            coeffs: [c64::default(); NCOEF],
            double_coeffs: [c64::default(); NCOEF],
        }
    }
}

/// Safe access to the full `R_n^m` table: returns zero for
/// `|m| > n`, which the source-gradient recurrences need at boundary
/// rows (e.g. `R_{n−1}^{n}` for the `R_n^{n−1}` derivative).
#[inline]
fn reg_at(reg: &[c64; NCOEF], n: usize, m: isize) -> c64 {
    if m.unsigned_abs() > n {
        c64::default()
    } else {
        reg[(n * n + n).wrapping_add_signed(m)]
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

    /// Accumulate one double-layer source: panel at `s` with density
    /// `f` and outward unit normal `n`. The source-gradient
    /// recurrences let us build `n · conj(∇_s R_n^m)` from the same
    /// regular-harmonic table as single-layer — one `regular_solid_harmonic`
    /// call covers both. Contribution to row `n = 0` is identically
    /// zero (`∇R_0^0 = 0`), so the inner loop starts at `n = 1`.
    pub(super) fn accumulate_dipole(&mut self, s: DVec3, f: f64, n: DVec3) {
        // why: the inner loop only ever reads `conj(R_{l-1}^*)`;
        // folding the 49 negations into one up-front pass strips
        // `.conj()` from the hot path.
        let reg: [c64; NCOEF] = regular_solid_harmonic(s - self.center).map(|c| c.conj());
        // why: n · ∇R_l^m in our convention collapses to
        //   n_minus · R*_{l-1,m-1} − n_plus · R*_{l-1,m+1} + n_z · R*_{l-1,m}
        // with n_plus = (n_x + i n_y)/2 and n_minus = conj(n_plus).
        let n_plus = c64::new(n.x * 0.5, n.y * 0.5);
        let n_minus = n_plus.conj();
        let n_z = c64::new(n.z, 0.0);
        let f_c = c64::new(f, 0.0);
        for l in 1..=P {
            let lm1 = l - 1;
            let l_sig = l as isize;
            for m in -l_sig..=l_sig {
                let r_lo = reg_at(&reg, lm1, m - 1);
                let r_hi = reg_at(&reg, lm1, m + 1);
                let r_mid = reg_at(&reg, lm1, m);
                let contrib = n_minus * r_lo - n_plus * r_hi + n_z * r_mid;
                let slot = (l * l + l).wrapping_add_signed(m);
                self.double_coeffs[slot] += contrib * f_c;
            }
        }
    }

    /// Evaluate `Σ_i q_i / (4π |r − s_i|)` at target `r`. Caller must
    /// guarantee `|r − center| > max_i |s_i − center|` for the
    /// truncated series to converge — debug-asserted only via the
    /// underlying `r² > 0` check inside [`irregular_solid_harmonic`].
    pub(super) fn evaluate_laplace(&self, r: DVec3) -> f64 {
        self.evaluate_laplace_pair(r).0
    }

    /// Double-layer Laplace evaluator: `Σ_b f_b · n_b · ∇_s G_0(r, s_b)`.
    pub(super) fn evaluate_double_laplace(&self, r: DVec3) -> f64 {
        self.evaluate_laplace_pair(r).1
    }

    /// Combined `(single, double)` evaluator. Shares the irregular
    /// solid-harmonic table + the `r − center` subtraction — the
    /// hot-path production caller (`traverse_bem`) always wants both.
    pub(super) fn evaluate_laplace_pair(&self, r: DVec3) -> (f64, f64) {
        let s = irregular_solid_harmonic(r - self.center);
        let mut sum_single = c64::default();
        let mut sum_double = c64::default();
        for ((&cs, &cd), &sn) in self
            .coeffs
            .iter()
            .zip(&self.double_coeffs)
            .zip(s.iter())
        {
            sum_single += cs * sn;
            sum_double += cd * sn;
        }
        // why: real-valued sources give M_n^{−m} = (−1)^m · conj(M_n^m);
        // paired with S_n^{−m} = (−1)^m · conj(S_n^m), cross-m terms
        // collapse to complex-conjugate pairs that cancel in the
        // imaginary part up to fp roundoff.
        debug_assert!(
            sum_single.im.abs() < 1e-6 * sum_single.re.abs().max(1.0),
            "Laplace SH single: imaginary residual {:e}",
            sum_single.im
        );
        debug_assert!(
            sum_double.im.abs() < 1e-6 * sum_double.re.abs().max(1.0),
            "Laplace SH double: imaginary residual {:e}",
            sum_double.im
        );
        (sum_single.re / FOUR_PI, sum_double.re / FOUR_PI)
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
        for (a, b) in self.double_coeffs.iter_mut().zip(&other.double_coeffs) {
            *a += b;
        }
    }

    /// Translate the expansion to a new centre via the M2M operator:
    /// ```text
    /// M'_n^m(c₂) = Σ_{k=0..n} Σ_{j=−k..k} conj(R_k^j(−δ)) · M_{n−k}^{m−j}(c₁)
    /// ```
    /// with `δ = c₂ − c₁`, derived from the regular-harmonic addition
    /// theorem `R_n^m(s − δ) = Σ R_{n−k}^{m−j}(s) · R_k^j(−δ)`.
    /// `O(P⁴)` complex multiplies per shift. The double-layer moments
    /// transform by the same kernel — `∂/∂s_α` commutes with the
    /// constant shift δ, so applying the translation theorem to
    /// `∇_s R_n^m(s − c₂)` lands the same per-coefficient coupling.
    pub(super) fn shifted_to(&self, new_center: DVec3) -> Self {
        let delta = new_center - self.center;
        // why: conjugate the shift table up front (49 negations) so the
        // quadruple inner loop sees a bare complex multiply. Also lets
        // us fold the `.conj()` out of the hottest path.
        let r_conj: [c64; NCOEF] = regular_solid_harmonic(-delta).map(|c| c.conj());
        let mut new_single = [c64::default(); NCOEF];
        let mut new_double = [c64::default(); NCOEF];
        for n in 0..=P {
            let n_sig = n as isize;
            let row_out = n * n + n;
            for m in -n_sig..=n_sig {
                let mut s_sum = c64::default();
                let mut d_sum = c64::default();
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
                        let src_slot = row_np.wrapping_add_signed(m - j);
                        s_sum += rc * self.coeffs[src_slot];
                        d_sum += rc * self.double_coeffs[src_slot];
                    }
                }
                let out_slot = row_out.wrapping_add_signed(m);
                new_single[out_slot] = s_sum;
                new_double[out_slot] = d_sum;
            }
        }
        Self {
            center: new_center,
            coeffs: new_single,
            double_coeffs: new_double,
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
        let mut sum = c64::default();
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

    /// Direct dipole sum `Σ_b f_b · n_b · ∇_s G_0(r, s_b)` for the
    /// Laplace kernel. Exact reference for the double-layer tests.
    fn direct_double_sum(sources: &[(DVec3, f64, DVec3)], target: DVec3) -> f64 {
        sources
            .iter()
            .map(|&(s, f, n)| {
                let d = target - s;
                let r = d.length();
                f * n.dot(d) / (FOUR_PI * r * r * r)
            })
            .sum()
    }

    #[test]
    fn sh_double_single_source_at_center_is_exact() {
        // With one source at the expansion centre, only Md_1^m
        // coefficients are nonzero; the truncated evaluator still
        // reproduces the full dipole field since ∇R_1^m spans
        // ∂/∂{x,y,z} R_0^0.
        let c = DVec3::new(1.0, 2.0, 3.0);
        let n = DVec3::new(0.3, -0.4, 0.866_025).normalize();
        let mut exp = ShExpansion {
            center: c,
            ..ShExpansion::default()
        };
        exp.accumulate_dipole(c, 2.5, n);
        let target = DVec3::new(10.0, 10.0, 10.0);
        let got = exp.evaluate_double_laplace(target);
        let expected = direct_double_sum(&[(c, 2.5, n)], target);
        assert!(
            (got - expected).abs() < 1e-14,
            "got={got}, expected={expected}"
        );
    }

    #[test]
    fn sh_double_far_field_decays_monotonically() {
        // Cluster of 10 point-dipoles in a small ball; relative error
        // must shrink with distance. At P = 6 the truncation envelope
        // is still (ε/R)^7.
        let sources: Vec<(DVec3, f64, DVec3)> = (0..10)
            .map(|i| {
                let t = i as f64 * 0.21;
                let s = DVec3::new(t.sin(), t.cos() * 0.8, (2.0 * t).sin()) * 0.5;
                let n = DVec3::new(t.cos(), -(t.sin()), (t * 3.0).sin()).normalize();
                (s, 1.0 + 0.1 * i as f64, n)
            })
            .collect();
        let mut exp = ShExpansion {
            center: DVec3::ZERO,
            ..ShExpansion::default()
        };
        for &(s, f, n) in &sources {
            exp.accumulate_dipole(s, f, n);
        }
        let mut prev = f64::INFINITY;
        for distance in [5.0_f64, 10.0, 20.0, 40.0] {
            let target = DVec3::new(distance, 0.3 * distance, -0.1 * distance);
            let approx = exp.evaluate_double_laplace(target);
            let exact = direct_double_sum(&sources, target);
            let rel = (approx - exact).abs() / exact.abs();
            assert!(rel < prev, "d={distance}: rel={rel:e}, prev={prev:e}");
            prev = rel;
        }
        assert!(prev < 1e-9, "far-field double error too large: {prev:e}");
    }

    #[test]
    fn sh_double_shift_matches_rebuild() {
        // M2M shift must preserve double-layer moments bit-for-bit.
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
        let mut at_c1 = ShExpansion {
            center: c1,
            ..ShExpansion::default()
        };
        let mut at_c2 = ShExpansion {
            center: c2,
            ..ShExpansion::default()
        };
        for &(s, f, n) in &sources {
            at_c1.accumulate_dipole(s, f, n);
            at_c2.accumulate_dipole(s, f, n);
        }
        let shifted = at_c1.shifted_to(c2);
        for (i, (&got, &want)) in shifted
            .double_coeffs
            .iter()
            .zip(&at_c2.double_coeffs)
            .enumerate()
        {
            let diff = (got - want).norm();
            let scale = 1.0 + want.norm();
            assert!(diff < 1e-11 * scale, "coeff {i}: diff={diff:e}");
        }
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
        // Both coefficient arrays must fold distributively — single-
        // layer moments and double-layer moments are independent
        // linear accumulators and fold visits both.
        let c = DVec3::new(0.5, -0.2, 1.0);
        let panels: Vec<(DVec3, f64, f64, DVec3)> = (0..10)
            .map(|i| {
                let t = i as f64 * 0.21;
                let pos = c + DVec3::new(t.sin(), t.cos(), (2.0 * t).sin()) * 0.1;
                let normal = DVec3::new(t.cos(), -t.sin(), (t + 0.3).sin()).normalize();
                (pos, 1.0 + 0.3 * i as f64, 0.5 + 0.2 * i as f64, normal)
            })
            .collect();
        let build = |subset: &[(DVec3, f64, f64, DVec3)]| {
            let mut exp = ShExpansion {
                center: c,
                ..ShExpansion::default()
            };
            for &(pos, q, f, nrm) in subset {
                exp.accumulate(pos, q);
                exp.accumulate_dipole(pos, f, nrm);
            }
            exp
        };
        let monolith = build(&panels);
        let (left, right) = panels.split_at(panels.len() / 2);
        let mut merged = build(left);
        merged.fold(&build(right));
        for (&got, &want) in merged.coeffs.iter().zip(&monolith.coeffs) {
            assert!((got - want).norm() < 1e-13, "single-layer fold mismatch");
        }
        for (&got, &want) in merged.double_coeffs.iter().zip(&monolith.double_coeffs) {
            assert!((got - want).norm() < 1e-13, "double-layer fold mismatch");
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
