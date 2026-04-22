//! Kirkwood analytical reaction-field potential with salt (κ > 0) for a
//! dielectric sphere with **point charges exterior to the sphere**.
//!
//! Derived by matching a Yukawa point-source expansion to a regular
//! interior Laplace expansion at `r = a`. For an exterior source with
//! `|r_i| > a` and a field point with `|r_j| > a`, the reaction-field
//! potential per unit source charge is
//!
//! ```text
//! φ_rf(r_j; r_i) = (κ / ε_out) · Σ_{n=0}^∞ (2n + 1) · k_n(κ r_i) k_n(κ r_j)
//!                  · [ ε_in n i_n(x) − ε_out x i_n'(x) ]
//!                  / [ ε_out x k_n'(x) − ε_in n k_n(x) ]
//!                  · P_n(cos γ_ij)
//! ```
//!
//! with `x = κ a` and `i_n`, `k_n` the modified spherical Bessel functions
//! of the first and second kind (see `bessel` module). At `κ → 0` each
//! term vanishes as O(κ³) for n ≥ 1 after scaling with k_n(κr) factors,
//! reproducing the κ = 0 Kirkwood formula; at large κ the exponential
//! decay of `k_n(κr)` dominates and `|φ_rf| → 0`.

use super::bessel::{i_deriv, i_series, k_deriv, k_series};
use glam::DVec3;

/// Reaction-field potential at `r_eval` due to a unit point source at
/// `r_source`, both exterior to a sphere of radius `a` centered at origin,
/// summed to `n_max` multipole terms.
pub fn reaction_field_potential_unit_source(
    r_source: [f64; 3],
    r_eval: [f64; 3],
    a: f64,
    eps_in: f64,
    eps_out: f64,
    kappa: f64,
    n_max: usize,
) -> f64 {
    let rs = DVec3::from(r_source);
    let re = DVec3::from(r_eval);
    let r_i = rs.length();
    let r_j = re.length();
    debug_assert!(
        r_i > a && r_j > a,
        "Kirkwood requires both points outside the sphere"
    );
    debug_assert!(kappa > 0.0, "use analytical::kirkwood for κ = 0");

    let cos_gamma = rs.dot(re) / (r_i * r_j);
    let x = kappa * a;

    let i_x = i_series(x, n_max);
    let k_x = k_series(x, n_max);
    let k_ri = k_series(kappa * r_i, n_max);
    let k_rj = k_series(kappa * r_j, n_max);

    let mut pn_minus_2 = 1.0_f64;
    let mut pn_minus_1 = cos_gamma;
    let mut sum = 0.0;
    for n in 0..=n_max {
        let p_n = match n {
            0 => 1.0,
            1 => cos_gamma,
            _ => {
                let nn = n as f64;
                let p = ((2.0 * nn - 1.0) * cos_gamma * pn_minus_1 - (nn - 1.0) * pn_minus_2) / nn;
                pn_minus_2 = pn_minus_1;
                pn_minus_1 = p;
                p
            }
        };

        let nf = n as f64;
        let i_n = i_x[n];
        let k_n_x = k_x[n];
        let ip = i_deriv(n, x, &i_x);
        let kp = k_deriv(n, x, &k_x);
        // why: B_n/q from BC matching at r = a (φ continuous,
        // ε·∂_r φ continuous), with the interior Laplace solution
        // D_n r^n P_n and exterior Yukawa response B_n k_n(κr) P_n.
        let numerator = eps_in * nf * i_n - eps_out * x * ip;
        let denominator = eps_out * x * kp - eps_in * nf * k_n_x;
        let coeff = numerator / denominator;

        let term = (2.0 * nf + 1.0) * k_ri[n] * k_rj[n] * coeff * p_n;

        // why: at small κa the individual k_n(κa), k_n(κr_i), k_n(κr_j)
        // explode for high n even though their ratio (which is what
        // survives after B_n substitution) stays finite. Once we see a
        // non-finite term, the series has outrun f64 range; further
        // terms would be noise. Stop.
        if !term.is_finite() {
            break;
        }
        sum += term;
    }

    kappa * sum / eps_out
}

/// Reaction-field contribution to the pairwise interaction energy
/// `W_ij = q_i q_j · φ_rf(r_j; r_i)` in reduced `e²/Å`.
#[allow(clippy::too_many_arguments)]
pub fn pair_reaction_energy(
    r_i: [f64; 3],
    q_i: f64,
    r_j: [f64; 3],
    q_j: f64,
    a: f64,
    eps_in: f64,
    eps_out: f64,
    kappa: f64,
    n_max: usize,
) -> f64 {
    let phi = reaction_field_potential_unit_source(r_i, r_j, a, eps_in, eps_out, kappa, n_max);
    q_i * q_j * phi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocity() {
        let a = 10.0;
        let r1 = [15.0, 0.0, 0.0];
        let r2 = [7.5, 7.5 * 3f64.sqrt(), 0.0];
        let phi_ij = reaction_field_potential_unit_source(r1, r2, a, 2.0, 80.0, 1.0 / 14.0, 40);
        let phi_ji = reaction_field_potential_unit_source(r2, r1, a, 2.0, 80.0, 1.0 / 14.0, 40);
        assert!((phi_ij - phi_ji).abs() / phi_ij.abs() < 1e-10);
    }

    #[test]
    fn screening_decreases_magnitude() {
        // Monotonic screening: |φ_rf(κ)| is a strictly decreasing function
        // of κ at physiological strengths.
        let a = 10.0;
        let r1 = [15.0, 0.0, 0.0];
        let r2 = [7.5, 7.5 * 3f64.sqrt(), 0.0];
        let phi_weak = reaction_field_potential_unit_source(r1, r2, a, 2.0, 80.0, 1.0 / 100.0, 40);
        let phi_mid = reaction_field_potential_unit_source(r1, r2, a, 2.0, 80.0, 1.0 / 14.0, 40);
        let phi_strong = reaction_field_potential_unit_source(r1, r2, a, 2.0, 80.0, 1.0 / 3.0, 40);
        assert!(phi_weak.abs() > phi_mid.abs());
        assert!(phi_mid.abs() > phi_strong.abs());
    }

    #[test]
    fn limit_small_kappa_approaches_unsalted_kirkwood() {
        use super::super::kirkwood::reaction_field_potential_unit_source as kirk0;
        let a = 10.0;
        let r1 = [15.0, 0.0, 0.0];
        let r2 = [7.5, 7.5 * 3f64.sqrt(), 0.0];
        let phi_salted = reaction_field_potential_unit_source(r1, r2, a, 2.0, 80.0, 1e-4, 60);
        let phi_unsalted = kirk0(r1, r2, a, 2.0, 80.0, 60);
        // With κ a = 1e-3, the salted formula should differ from the
        // κ = 0 formula by less than a percent — they describe the same
        // limit.
        let rel_err = (phi_salted - phi_unsalted).abs() / phi_unsalted.abs();
        assert!(rel_err < 1e-2, "rel err {rel_err:e} (κ→0 consistency)");
    }
}
