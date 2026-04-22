//! Kirkwood 1934 analytical reaction-field potential for a dielectric sphere
//! with **charges interior to the sphere** (the production MC-titration case).
//!
//! Derived by matching an interior Coulomb + harmonic expansion to an
//! exterior decaying expansion at `r = a`. For two interior charges with
//! `|r_i| < a` and `|r_j| < a`, the reaction-field potential per unit
//! source charge in reduced units is
//!
//! ```text
//! φ_rf(r_j; r_i) = (1/ε_in) · Σ_{n=0}^∞  [ (n+1)(ε_in − ε_out) ]
//!                                        / [ n·ε_in + (n+1)·ε_out ]
//!                                        · (r_i r_j)^n / a^(2n+1)
//!                                        · P_n(cos γ_ij)
//! ```
//!
//! This is the classical Kirkwood 1934 formula (the "inside" case) —
//! same functional form as Tanford–Kirkwood 1957 eq (7) once the `b/D_i`
//! factorization is undone. For `ε_in < ε_out` (protein in water) the
//! coefficient is negative, so like-charge pairs inside the cavity see a
//! stabilizing reaction field — polarization of the surrounding solvent
//! partially screens their interaction. κ has no effect on the interior
//! solution (Laplace only).
//!
//! All in reduced units (charge in `e`, distance in Å, φ in e/Å). Sign
//! convention: `φ_rf = φ_total − φ_Coulomb_in_ε_in` (the reaction-field
//! correction, not the total potential).

use super::legendre::legendre_series;
use glam::DVec3;

/// Reaction-field potential at `r_eval` due to a unit point source at
/// `r_source`, both interior to a sphere of radius `a` centered at origin,
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
        r_i < a && r_j < a,
        "Kirkwood-inside requires both points strictly inside the sphere"
    );

    let cos_gamma = rs.dot(re) / (r_i * r_j);
    let p = legendre_series(cos_gamma, n_max);

    let rij = r_i * r_j;
    let a2 = a * a;
    let mut rij_pow = 1.0_f64;
    let mut a_pow = a; // a^(2n+1) with n=0 → a.

    let mut sum = 0.0;
    for (n, &p_n) in p.iter().enumerate() {
        let nf = n as f64;
        let coeff = (nf + 1.0) * (eps_in - eps_out) / (nf * eps_in + (nf + 1.0) * eps_out);
        sum += coeff * rij_pow / a_pow * p_n;

        rij_pow *= rij;
        a_pow *= a2;
    }

    sum / eps_in
}

/// Reaction-field contribution to the pairwise interaction energy
/// `W_ij = q_i q_j · φ_rf(r_j; r_i)` (reduced `e²/Å`).
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
    fn reciprocity() {
        let a = 10.0;
        let r1 = [5.0, 0.0, 0.0];
        let r2 = [3.0, 4.0, 0.0]; // |r2| = 5
        let phi_ij = reaction_field_potential_unit_source(r1, r2, a, 2.0, 80.0, 40);
        let phi_ji = reaction_field_potential_unit_source(r2, r1, a, 2.0, 80.0, 40);
        assert!((phi_ij - phi_ji).abs() / phi_ij.abs() < 1e-12);
    }

    #[test]
    fn no_contrast_gives_zero() {
        // ε_in = ε_out ⇒ no dielectric boundary effect.
        let a = 10.0;
        let r1 = [5.0, 0.0, 0.0];
        let r2 = [0.0, 5.0, 0.0];
        let phi = reaction_field_potential_unit_source(r1, r2, a, 80.0, 80.0, 40);
        assert!(phi.abs() < 1e-14);
    }

    #[test]
    fn like_charges_see_stabilizing_reaction_field() {
        // For ε_in < ε_out, like-charge pair interaction is screened
        // by boundary polarization ⇒ W_rf is negative.
        let a = 10.0;
        let r1 = [5.0, 0.0, 0.0];
        let r2 = [3.0, 4.0, 0.0];
        let w = pair_reaction_energy(r1, 1.0, r2, 1.0, a, 2.0, 80.0, 40);
        assert!(
            w < 0.0,
            "expected W_rf < 0 for like-charge inside pair, got {w}"
        );
    }
}
