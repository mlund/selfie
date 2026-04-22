//! Panel-integrated kernels K₀ and K₀' for centroid collocation.
//!
//! K₀(a, b)  = ∫_{T_b} 1/(4π|c_a − s'|)            dS'
//! K₀'(a, b) = ∫_{T_b} (c_a − s')·n_b / (4π|c_a − s'|³) dS'
//!
//! Two regimes:
//! - **Off-diagonal** (a ≠ b): smooth integrand → 3-point barycentric
//!   Gauss rule (degree 2, O(h³) per panel for polynomial integrands).
//!   Needed because near-neighbour panels have a 1/r integrand that varies
//!   fast enough to defeat 1-point centroid quadrature.
//! - **Self-panel (a = b)**: K₀ is weakly singular (1/r). Closed-form
//!   Wilton–Rao–Glisson 1984 evaluation — three logs per triangle, no Duffy
//!   transform, no recursion. K₀' PV is 0 for a flat panel at its own
//!   centroid (displacement `c − s'` lies in the panel plane, so its dot
//!   product with `n_b` vanishes pointwise); the ½I jump in the BIE is a
//!   separate matrix contribution, not a diagonal of K₀'.

use glam::DVec3;

const FOUR_PI: f64 = 4.0 * core::f64::consts::PI;

/// Off-diagonal K₀ via 3-point symmetric Gauss (barycentrics
/// `(2/3, 1/6, 1/6)` + cyclic permutations, each weight 1/3).
pub(crate) fn k0_off(observer: DVec3, tri: [DVec3; 3], ab: f64) -> f64 {
    let mut acc = 0.0;
    for p in gauss3_points(tri) {
        let r = (observer - p).length();
        acc += 1.0 / (FOUR_PI * r);
    }
    acc * ab / 3.0
}

/// Off-diagonal K₀': 3-point symmetric Gauss rule. Integrand
/// `(c_a − s')·n_b / (4π |c_a − s'|³)`.
pub(crate) fn k0_prime_off(observer: DVec3, tri: [DVec3; 3], nb: DVec3, ab: f64) -> f64 {
    let mut acc = 0.0;
    for p in gauss3_points(tri) {
        let d = observer - p;
        let r = d.length();
        acc += d.dot(nb) / (FOUR_PI * r * r * r);
    }
    acc * ab / 3.0
}

/// Barycentric 3-point rule suited for smooth integrands on a triangle.
/// Degree-of-exactness 2.
pub(crate) fn gauss3_points(tri: [DVec3; 3]) -> [DVec3; 3] {
    // Barycentrics (2/3, 1/6, 1/6) + cyclic permutations.
    const A: f64 = 2.0 / 3.0;
    const B: f64 = 1.0 / 6.0;
    [
        tri[0] * A + tri[1] * B + tri[2] * B,
        tri[0] * B + tri[1] * A + tri[2] * B,
        tri[0] * B + tri[1] * B + tri[2] * A,
    ]
}

/// Self-panel K₀: Wilton–Rao–Glisson closed form for
/// `∫_T 1/(4π|p − s'|) dS'` with `p` = in-plane observer (here, centroid).
///
/// Derivation reference: Wilton, Rao, Glisson, *IEEE TAP* 1984
/// (https://doi.org/10.1109/TAP.1984.1143193).
///
/// Algorithm: for each edge k with endpoints `v_k, v_{k+1}`, sum
/// `d_k · ln( (R_k⁺ + l_k⁺) / (R_k⁻ + l_k⁻) )` where
/// - `R_k^±` = |p − v_k^±|
/// - `l_k^±` = signed distance from projection of `p` onto edge line to
///   the endpoints (positive along edge direction)
/// - `d_k` = perpendicular in-plane distance from `p` to the edge line.
pub(crate) fn k0_self_wrg(v: [DVec3; 3], p: DVec3) -> f64 {
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
        let got = k0_self_wrg(v, centroid);

        assert!(
            (got - expected).abs() < 1e-12,
            "WRG = {}, expected {}, diff {:e}",
            got,
            expected,
            (got - expected).abs()
        );
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

        let three_point = k0_off(observer, v, area);
        let refined = refine_integrate_g0_off(v, observer, 50);
        let rel_err = (three_point - refined).abs() / refined.abs();
        // why: 3-point Gauss is exact for degree-2 polynomial integrands,
        // O(h³) for smooth non-polynomial ones. Far-field test ≪ 1e-5.
        assert!(rel_err < 1e-5, "rel err {:e}", rel_err);
    }
}
