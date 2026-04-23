//! Panel-integrated kernels K_κ and K_κ' for centroid collocation.
//!
//! K_κ(a, b)  = ∫_{T_b} G_κ(c_a, s')                  dS'
//! K_κ'(a, b) = ∫_{T_b} ∂_{n_b, s'} G_κ(c_a, s')       dS'
//!
//! with the Yukawa (screened-Coulomb) Green's function
//! `G_κ(r, s) = exp(−κ|r − s|) / (4π|r − s|)`. `κ = 0` recovers the plain
//! Coulomb kernel `G_0 = 1/(4πr)` so the same routines cover the Laplace
//! top-block of the Juffer system and the Yukawa bottom-block.
//!
//! Two regimes:
//! - **Off-diagonal** (a ≠ b): smooth integrand → 3-point barycentric
//!   Gauss rule (degree 2, O(h³) per panel for polynomial integrands).
//!   Needed because near-neighbour panels have a 1/r integrand that varies
//!   fast enough to defeat 1-point centroid quadrature.
//! - **Self-panel (a = b)**: K_κ has a weak 1/r singularity. For κ = 0 we
//!   use the Wilton–Rao–Glisson 1984 closed form (three logs per triangle,
//!   no Duffy, no recursion). For κ > 0 we split G_κ = G_0 + (G_κ − G_0);
//!   the singular part is WRG, the bounded smooth correction is evaluated
//!   with 3-point Gauss (limit at r = 0 is −κ/(4π)). K_κ' PV is 0 for a
//!   flat panel at its own centroid regardless of κ: `(c − s')·n_b`
//!   vanishes pointwise for s' in the panel plane. The ½I jump in the BIE
//!   is a separate matrix contribution, not a diagonal of K_κ'.

use glam::DVec3;

const FOUR_PI: f64 = 4.0 * core::f64::consts::PI;

/// Barycentric 3-point Gauss rule (degree 2) on a triangle, with
/// explicit weights so the caller scales by `area / (4π)` rather than
/// `area / (3·4π)` — keeping the weight convention identical to the
/// 7-point rule below.
pub(super) fn gauss3_points(tri: [DVec3; 3]) -> [(DVec3, f64); 3] {
    // Barycentrics (2/3, 1/6, 1/6) + cyclic permutations.
    const A: f64 = 2.0 / 3.0;
    const B: f64 = 1.0 / 6.0;
    const W: f64 = 1.0 / 3.0;
    [
        (tri[0] * A + tri[1] * B + tri[2] * B, W),
        (tri[0] * B + tri[1] * A + tri[2] * B, W),
        (tri[0] * B + tri[1] * B + tri[2] * A, W),
    ]
}

/// Dunavant degree-5 rule on a reference triangle, mapped to world-space
/// via barycentric interpolation. The 7 barycentric triples + weights
/// below are the standard Stroud/Dunavant set (weights sum to 1).
fn gauss7_points(tri: [DVec3; 3]) -> [(DVec3, f64); 7] {
    const T: f64 = 1.0 / 3.0;
    const W0: f64 = 9.0 / 40.0;
    const A1: f64 = 0.797_426_985_353_087_3;
    const B1: f64 = 0.101_286_507_323_456_3;
    const W1: f64 = 0.125_939_180_544_827_1;
    const A2: f64 = 0.059_715_871_789_769_8;
    const B2: f64 = 0.470_142_064_105_115_1;
    const W2: f64 = 0.132_394_152_788_506_2;
    let p = |a: f64, b: f64, c: f64| tri[0] * a + tri[1] * b + tri[2] * c;
    [
        (p(T, T, T), W0),
        (p(A1, B1, B1), W1),
        (p(B1, A1, B1), W1),
        (p(B1, B1, A1), W1),
        (p(A2, B2, B2), W2),
        (p(B2, A2, B2), W2),
        (p(B2, B2, A2), W2),
    ]
}

/// Off-diagonal K_κ and K_κ' at once via 3-point Gauss (degree 2,
/// O(h³) per panel). Pass `kappa = 0.0` for the Laplace kernel G_0.
pub fn k_and_kprime_off(
    observer: DVec3,
    tri: [DVec3; 3],
    nb: DVec3,
    ab: f64,
    kappa: f64,
) -> (f64, f64) {
    accumulate_kernel(&gauss3_points(tri), observer, nb, ab, kappa)
}

/// 7-point Dunavant version of [`k_and_kprime_off`] for near-singular
/// pairs. Each near-field entry computed with the coarser 3-point rule
/// has enough O(h³) error to push GMRES iteration counts sharply up on
/// heterogeneous protein meshes; 7-point (degree 5, O(h⁶)) restores
/// pygbe-level convergence — their `K_fine = 19` parameter serves the
/// same purpose.
pub fn k_and_kprime_near(
    observer: DVec3,
    tri: [DVec3; 3],
    nb: DVec3,
    ab: f64,
    kappa: f64,
) -> (f64, f64) {
    accumulate_kernel(&gauss7_points(tri), observer, nb, ab, kappa)
}

/// Shared quadrature accumulator for both rules. The κ = 0 branch
/// skips the `exp()` and `κr + 1` factors — the Laplace block of the
/// Juffer system has κ = 0 by construction, and profiling measured
/// `libm::exp` at ~39 % of solve wall time before this branch.
fn accumulate_kernel(
    points: &[(DVec3, f64)],
    observer: DVec3,
    nb: DVec3,
    ab: f64,
    kappa: f64,
) -> (f64, f64) {
    let mut acc_k = 0.0;
    let mut acc_kp = 0.0;
    if kappa == 0.0 {
        for &(p, w) in points {
            let d = observer - p;
            let r = d.length();
            acc_k += w / r;
            acc_kp += w * d.dot(nb) / (r * r * r);
        }
    } else {
        for &(p, w) in points {
            let d = observer - p;
            let r = d.length();
            let exp_kr = (-kappa * r).exp();
            acc_k += w * exp_kr / r;
            acc_kp += w * kappa.mul_add(r, 1.0) * exp_kr * d.dot(nb) / (r * r * r);
        }
    }
    let s = ab / FOUR_PI;
    (acc_k * s, acc_kp * s)
}

/// Self-panel K_κ = ∫_T G_κ(p, s') dS' at the panel centroid `p`.
///
/// Splits G_κ = G_0 + (G_κ − G_0). The singular part is Wilton–Rao–Glisson;
/// the bounded correction `(exp(−κr) − 1) / (4π r)` — finite with limit
/// −κ/(4π) as r → 0 — is integrated with 3-point Gauss.
pub fn k_self(tri: [DVec3; 3], centroid: DVec3, kappa: f64) -> f64 {
    let k0 = wrg_g0_self(tri, centroid);
    if kappa == 0.0 {
        return k0;
    }
    let area = 0.5 * (tri[1] - tri[0]).cross(tri[2] - tri[0]).length();
    let mut corr = 0.0;
    for (p, w) in gauss3_points(tri) {
        let r = (centroid - p).length();
        // why: (exp(−κr) − 1)/r is analytic at r = 0 with limit −κ; guard
        // the Gauss point that may coincide with the centroid (it doesn't
        // for the (2/3, 1/6, 1/6) rule — centroid has barycentric (1/3)³ —
        // but f64 cancellation could still produce tiny r).
        let integrand = if r > 1e-14 {
            (-kappa * r).exp_m1() / r
        } else {
            -kappa
        };
        corr += w * integrand;
    }
    k0 + corr * area / FOUR_PI
}

/// Wilton–Rao–Glisson closed form for `∫_T 1/(4π|p − s'|) dS'` with `p` in
/// the panel plane (here, the centroid).
///
/// Reference: Wilton, Rao, Glisson, *IEEE TAP* 1984
/// (https://doi.org/10.1109/TAP.1984.1143193).
///
/// Algorithm: for each edge k with endpoints `v_k, v_{k+1}`, sum
/// `d_k · ln( (R_k⁺ + l_k⁺) / (R_k⁻ + l_k⁻) )` where
/// - `R_k^±` = |p − v_k^±|
/// - `l_k^±` = signed distance from projection of `p` onto edge line to
///   the endpoints (positive along edge direction)
/// - `d_k` = perpendicular in-plane distance from `p` to the edge line.
fn wrg_g0_self(v: [DVec3; 3], p: DVec3) -> f64 {
    // Triangle plane normal (unit). A self-panel lookup expects `p` in the
    // plane of the triangle; we still project to be robust against f64 drift.
    let e01 = v[1] - v[0];
    let e02 = v[2] - v[0];
    let n = e01.cross(e02).normalize();

    // In-plane observer.
    let p_plane = p - n * (p - v[0]).dot(n);

    let mut acc = 0.0;
    for k in 0..3 {
        let a = v[k];
        let b = v[(k + 1) % 3];
        let edge = b - a;
        let edge_len = edge.length();
        let u = edge / edge_len;

        // Perpendicular-in-plane direction from the edge, pointing away from
        // the edge toward the opposite vertex side (sign matters; d_k is a
        // signed quantity in some derivations, but here we use |d_k| and the
        // log's sign naturally handles inside/outside).
        let m = n.cross(u);

        // Signed in-plane perpendicular distance from p to the edge line.
        // |p - a| projected onto m.
        let d_k = (p_plane - a).dot(m);

        // Signed along-edge distances from p's projection to the endpoints.
        let l_plus = (b - p_plane).dot(u);
        let l_minus = (a - p_plane).dot(u);

        // Distances from p to endpoints.
        let r_plus = (b - p_plane).length();
        let r_minus = (a - p_plane).length();

        // why: when p is collinear with the edge, r + l → 0 and the log
        // diverges. This can't happen for p = centroid of a non-degenerate
        // triangle (centroid is strictly interior). Guard it as a debug
        // assertion for the BEM path; callers with unusual geometry should
        // use a higher-order routine.
        debug_assert!(
            r_plus + l_plus > 0.0 && r_minus + l_minus > 0.0,
            "WRG log argument non-positive; observer collinear with edge?"
        );

        acc += d_k * ((r_plus + l_plus) / (r_minus + l_minus)).ln();
    }

    acc / FOUR_PI
}

#[cfg(test)]
mod tests {
    use super::*;

    fn barycentric(v: [DVec3; 3], s: f64, t: f64) -> DVec3 {
        v[0] * (1.0 - s - t) + v[1] * s + v[2] * t
    }

    /// Sub-integrate 1/(4πr) over a triangle using barycentric subdivision,
    /// **excluding** the single sub-triangle that contains the observer. For
    /// an off-centroid observer this converges to the full integral; for
    /// the centroid case we compare WRG to a known closed form instead
    /// (see [`wrg_diagonal_matches_equilateral_closed_form`]).
    fn refine_integrate_g0_off(v: [DVec3; 3], p: DVec3, n_per_edge: usize) -> f64 {
        let mut sum = 0.0;
        let n = n_per_edge as f64;
        let area_total = 0.5 * (v[1] - v[0]).cross(v[2] - v[0]).length();
        for i in 0..n_per_edge {
            for j in 0..(n_per_edge - i) {
                let a = barycentric(v, (i as f64) / n, (j as f64) / n);
                let b = barycentric(v, ((i + 1) as f64) / n, (j as f64) / n);
                let c = barycentric(v, (i as f64) / n, ((j + 1) as f64) / n);
                let centroid = (a + b + c) / 3.0;
                let sub_area = area_total / (n * n);
                let r = (p - centroid).length();
                sum += sub_area / (FOUR_PI * r);

                if j + i + 1 < n_per_edge {
                    let d = barycentric(v, ((i + 1) as f64) / n, ((j + 1) as f64) / n);
                    let centroid2 = (b + d + c) / 3.0;
                    let r2 = (p - centroid2).length();
                    sum += sub_area / (FOUR_PI * r2);
                }
            }
        }
        sum
    }

    #[test]
    fn wrg_diagonal_matches_equilateral_closed_form() {
        // For an equilateral triangle with side s, observer at centroid,
        // each of the 3 WRG edges contributes d · ln((2+√3)/(2−√3))/(4π)
        // where d = s/(2√3) is the centroid-to-edge perpendicular. Symbolic
        // simplification: I = (3·s/(2√3) · ln(7+4√3)) / (4π)
        //                   = (s·√3/2 · ln(7+4√3)) / (4π).
        // For s = 1 this is ≈ 0.18151923565714137 — a closed-form target.
        let s = 1.0_f64;
        let v = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(s, 0.0, 0.0),
            DVec3::new(s * 0.5, s * (3f64.sqrt() / 2.0), 0.0),
        ];
        let centroid = (v[0] + v[1] + v[2]) / 3.0;

        let expected = (s * (3f64.sqrt() / 2.0) * (7.0 + 4.0 * 3f64.sqrt()).ln())
            / (4.0 * core::f64::consts::PI);
        let got = k_self(v, centroid, 0.0);

        assert!(
            (got - expected).abs() < 1e-12,
            "WRG = {}, expected {}, diff {:e}",
            got,
            expected,
            (got - expected).abs()
        );
    }

    #[test]
    fn k_self_yukawa_reduces_to_wrg_at_kappa_zero() {
        let v = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.5, 3f64.sqrt() / 2.0, 0.0),
        ];
        let centroid = (v[0] + v[1] + v[2]) / 3.0;
        let k0 = k_self(v, centroid, 0.0);
        let kk = k_self(v, centroid, 1e-12);
        assert!((k0 - kk).abs() / k0.abs() < 1e-10);
    }

    #[test]
    fn k_self_yukawa_is_smaller_than_coulomb() {
        // exp(−κr) ≤ 1 ⇒ K_κ_self ≤ K_0_self for κ > 0.
        let v = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.5, 3f64.sqrt() / 2.0, 0.0),
        ];
        let centroid = (v[0] + v[1] + v[2]) / 3.0;
        let k0 = k_self(v, centroid, 0.0);
        let kk = k_self(v, centroid, 1.0);
        assert!(kk < k0 && kk > 0.0);
    }

    #[test]
    fn off_diagonal_k0_matches_refined_quadrature() {
        let v = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.5, 0.866, 0.0),
        ];
        let area = 0.5 * (v[1] - v[0]).cross(v[2] - v[0]).length();
        let observer = DVec3::new(5.0, 5.0, 10.0);

        let (three_point, _) = k_and_kprime_off(observer, v, DVec3::Z, area, 0.0);
        let refined = refine_integrate_g0_off(v, observer, 50);
        let rel_err = (three_point - refined).abs() / refined.abs();
        // why: 3-point Gauss is exact for degree-2 polynomial integrands,
        // O(h³) for smooth non-polynomial ones. Far-field test ≪ 1e-5.
        assert!(rel_err < 1e-5, "rel err {:e}", rel_err);
    }
}
