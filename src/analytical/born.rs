//! Closed-form reaction-field potential and self-energy for a point charge
//! at the **centre** of a dielectric sphere.
//!
//! By spherical symmetry every multipole beyond `n = 0` vanishes, so the
//! full Kirkwood / Tanford–Kirkwood series collapses to a single term
//! with no Bessel-function machinery required. Two limits:
//!
//! - **κ = 0 (classical Born)**. Solving Laplace inside and matching BC at
//!   `r = a`:
//!   ```text
//!   φ_rf(0) = (q / a) · (1/ε_out − 1/ε_in)
//!   U_Born  = (q² / 2 a) · (1/ε_out − 1/ε_in)
//!   ```
//!   For `ε_in < ε_out` this is negative — solvent stabilises the interior
//!   charge.
//!
//! - **κ > 0 (Debye–Hückel)**. Exterior is a decaying Yukawa response
//!   `E_0 · exp(−κr)/r`; matching at `r = a` gives
//!   ```text
//!   φ_rf(0) = (q / a) · [ 1/(ε_out · (1 + κa)) − 1/ε_in ]
//!   ```
//!   Reduces to the Born form as `κ → 0`.
//!
//! Reduced units: distances in Å, charges in `e`, φ in e/Å, energies in
//! `e²/Å`.

/// Reaction-field potential at the sphere centre per unit source charge
/// (reduced units e/Å). Valid for any `κ ≥ 0`. The source is placed at
/// the origin; the field point is also the origin.
pub fn reaction_field_at_center(a: f64, eps_in: f64, eps_out: f64, kappa: f64) -> f64 {
    (1.0 / (eps_out * (1.0 + kappa * a)) - 1.0 / eps_in) / a
}

/// Born self-energy of a point charge `q` at the sphere centre
/// (reduced units e²/Å). For like values of `ε_in < ε_out` this is
/// negative (solvation stabilises the interior charge).
pub fn born_self_energy(q: f64, a: f64, eps_in: f64, eps_out: f64, kappa: f64) -> f64 {
    0.5 * q * q * reaction_field_at_center(a, eps_in, eps_out, kappa)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debye_huckel_reduces_to_born_at_kappa_zero() {
        let a = 10.0;
        let eps_in = 2.0;
        let eps_out = 80.0;
        let born = reaction_field_at_center(a, eps_in, eps_out, 0.0);
        let dh = reaction_field_at_center(a, eps_in, eps_out, 1e-8);
        assert!((born - dh).abs() / born.abs() < 1e-7);
    }

    #[test]
    fn born_matches_hand_calculation() {
        // q = 1, a = 10, ε_in = 2, ε_out = 80.
        // φ_rf(0) = (1/10) · (1/80 − 1/2) = 0.1 · (-0.4875) = -0.04875 e/Å.
        let phi = reaction_field_at_center(10.0, 2.0, 80.0, 0.0);
        assert!((phi - (-0.04875)).abs() < 1e-14);
    }

    #[test]
    fn salt_makes_born_more_stabilizing() {
        // Adding salt to the exterior screens the charge further → more
        // negative φ_rf at the centre (more negative self-energy).
        let a = 10.0;
        let dry = reaction_field_at_center(a, 2.0, 80.0, 0.0);
        let wet = reaction_field_at_center(a, 2.0, 80.0, 1.0 / 14.0);
        assert!(wet < dry, "expected wet ({wet}) < dry ({dry})");
    }
}
