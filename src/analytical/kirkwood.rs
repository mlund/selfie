//! Kirkwood-style analytical reaction-field potential for a dielectric sphere
//! in κ = 0 with **charges exterior to the sphere** (the case we actually
//! validate against — see bem_pb_plan.md §0 "All point charges live in Ω⁺").
//!
//! Derived by matching a point charge's exterior multipole expansion to a
//! regular interior expansion at r = a using φ continuous and
//! ε·∂_r φ continuous. Reference: standard boundary-value problem as in
//! Jackson §4.3 and Bottcher 1952; the specific closed form for
//! outside-sphere charges is:
//!
//! ```text
//! φ_rf(r_j; r_i) = (1/ε_out) · Σ_{n=0}^∞  [ n·(ε_out − ε_in) ]
//!                                         / [ ε_out·(n·ε_in + (n+1)·ε_out) ]
//!                                         · a^(2n+1) / (r_i r_j)^(n+1)
//!                                         · P_n(cos γ_ij)
//! ```
//!
//! Note the `n` (not `n+1`) prefactor in the numerator and the sign,
//! differing from bem_pb_plan.md §3 which appears to transcribe the
//! *inside-charge* Kirkwood 1934 original; the exterior-charge case has
//! B_0 = 0 (neutral sphere carries no induced monopole with an exterior
//! source), which this formula respects.
//!
//! All in reduced units (charge in `e`, distance in Å, φ in e/Å).
//!
//! **Sign convention**: this is φ_total − φ_Coulomb_in_ε_out — i.e. the
//! reaction-field *correction*, not the total potential. Multiplied by a
//! target charge q_j gives the reaction-field contribution to W_ij in e²/Å.

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

    let cos_gamma = rs.dot(re) / (r_i * r_j);

    // why: running products for a^(2n+1) and (r_i·r_j)^(n+1) replace O(log n)
    // `powi` calls per iteration with O(1), and the two-term Legendre
    // recurrence stays stable up to n ~ 10⁴. n_max ≈ 30 suffices for the
    // plan's r/a = 1.2 geometry where terms decay like (a/r)^(2n).
    let a2 = a * a;
    let rij = r_i * r_j;
    let mut a_pow = a;
    let mut rij_pow = rij;

    let mut pn_minus_2 = 1.0_f64;
    let mut pn_minus_1 = cos_gamma;
    let mut sum = 0.0;
    for n in 0..=n_max {
        let p_n = match n {
            0 => 1.0,
            1 => cos_gamma,
            _ => {
                let nn = n as f64;
                // (n) · P_n(x) = (2n−1) · x · P_{n−1}(x) − (n−1) · P_{n−2}(x)
                let p = ((2.0 * nn - 1.0) * cos_gamma * pn_minus_1 - (nn - 1.0) * pn_minus_2) / nn;
                pn_minus_2 = pn_minus_1;
                pn_minus_1 = p;
                p
            }
        };

        // why: B_n/q from BC matching (φ and ε·∂_r φ continuous) for a
        // point charge outside a dielectric sphere — the outside-charge
        // case (not the inside-charge Kirkwood 1934 original). The 1/ε_out
        // prefactor appears once, factored out of the loop below.
        let nf = n as f64;
        let coeff = nf * (eps_out - eps_in) / (nf * eps_in + (nf + 1.0) * eps_out);
        sum += coeff * (a_pow / rij_pow) * p_n;

        a_pow *= a2;
        rij_pow *= rij;
    }

    sum / eps_out
}

/// Reaction-field contribution to the pairwise interaction energy
/// `W_ij = q_j · φ_rf(r_j; r_i) · q_i` (symmetric; Kirkwood series
/// already includes the `1/ε_out` factor, so this is the direct reduced
/// `e²/Å` energy — no `k_e`).
#[allow(clippy::too_many_arguments)]
pub fn pair_reaction_energy(
    r_i: [f64; 3],
    q_i: f64,
    r_j: [f64; 3],
    q_j: f64,
    a: f64,
    eps_in: f64,
    eps_out: f64,
    n_max: usize,
) -> f64 {
    let phi = reaction_field_potential_unit_source(r_i, r_j, a, eps_in, eps_out, n_max);
    q_i * q_j * phi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_study_converges_by_30_terms() {
        // Plan geometry: r = 1.2 a, angle 60°, ε_in=2, ε_out=80.
        let a = 10.0;
        let r = 1.2 * a;
        let r1 = [r, 0.0, 0.0];
        let r2 = [r * 0.5, r * (3f64.sqrt() / 2.0), 0.0]; // 60° in plane

        let e_in = 2.0;
        let e_out = 80.0;

        let w30 = pair_reaction_energy(r1, 1.0, r2, 1.0, a, e_in, e_out, 30);
        let w100 = pair_reaction_energy(r1, 1.0, r2, 1.0, a, e_in, e_out, 100);

        let rel_err = (w30 - w100).abs() / w100.abs();
        // why: geometric factor (a/r)^n with r/a=1.2 means term n contributes
        // roughly 0.69^n — by n=30 residual ~ 10^-5.
        assert!(
            rel_err < 1e-5,
            "Kirkwood not converged by n=30: rel {:e}",
            rel_err
        );
    }

    #[test]
    fn reciprocity_phi_rf_is_symmetric_in_source_and_field() {
        // φ_rf(r_j; r_i) from a unit source at r_i evaluated at r_j must
        // equal φ_rf(r_i; r_j) (symmetric Green's function). This is the
        // strongest self-consistency check that doesn't require knowing the
        // sign of the answer a priori.
        let a = 10.0;
        let r1 = [12.0, 0.0, 0.0];
        let r2 = [6.0, 6.0 * 3f64.sqrt(), 0.0];
        let phi_ij = reaction_field_potential_unit_source(r1, r2, a, 2.0, 80.0, 40);
        let phi_ji = reaction_field_potential_unit_source(r2, r1, a, 2.0, 80.0, 40);
        assert!(
            (phi_ij - phi_ji).abs() / phi_ij.abs() < 1e-12,
            "φ_rf not symmetric: {} vs {}",
            phi_ij,
            phi_ji
        );
    }
}
