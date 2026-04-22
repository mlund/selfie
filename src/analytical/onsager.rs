//! Onsager (1936) reaction-field / solvation energy of a point dipole at
//! the centre of a dielectric cavity of radius `a`.
//!
//! Matching a dipole potential `μ·cos θ / (ε_in·r²)` inside with a decaying
//! `E·cos θ / r²` outside at `r = a` gives a uniform interior reaction
//! field of magnitude
//!
//! ```text
//! R = 2·μ·(ε_out − ε_in) / [ε_in · a³ · (2·ε_out + ε_in)]
//! ```
//!
//! parallel to the dipole. The corresponding solvation (work to move the
//! dipole from vacuum into the cavity) is
//!
//! ```text
//! U_solv = -½ · μ · R = -μ² · (ε_out − ε_in) / [ε_in · a³ · (2·ε_out + ε_in)]
//! ```
//!
//! Reduces to Onsager's original `f = 2(ε−1)/(a³(2ε+1))` when ε_in = 1.
//! Reduced units: μ in e·Å, a in Å, energy in e²/Å.

/// Reaction-field magnitude acting on a point dipole `μ` (e·Å) at the
/// centre of a sphere of radius `a` (Å), interior ε_in, exterior ε_out.
/// Output is the magnitude of the uniform interior reaction field in
/// reduced units (e/Å², i.e. field magnitude per unit dipole moment
/// integrated out).
pub fn reaction_field_inside(mu: f64, a: f64, eps_in: f64, eps_out: f64) -> f64 {
    2.0 * mu * (eps_out - eps_in) / (eps_in * a.powi(3) * (2.0 * eps_out + eps_in))
}

/// Solvation free energy of a point dipole `μ` at the sphere centre in
/// reduced units (e²/Å). Negative for ε_in < ε_out — the polarisable
/// exterior stabilises the dipole.
pub fn dipole_solvation_energy(mu: f64, a: f64, eps_in: f64, eps_out: f64) -> f64 {
    -0.5 * mu * reaction_field_inside(mu, a, eps_in, eps_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_onsager_vacuum_cavity_limit() {
        // Original Onsager 1936 formula: f = 2(ε − 1) / [a³·(2ε + 1)]
        // for a dipole in a vacuum cavity (ε_in = 1) in medium ε.
        let a = 10.0;
        let eps = 80.0;
        let r_us = reaction_field_inside(1.0, a, 1.0, eps);
        let r_onsager = 2.0 * (eps - 1.0) / (a.powi(3) * (2.0 * eps + 1.0));
        assert!((r_us - r_onsager).abs() < 1e-14);
    }

    #[test]
    fn no_contrast_gives_zero() {
        let u = dipole_solvation_energy(1.0, 10.0, 80.0, 80.0);
        assert!(u.abs() < 1e-14);
    }

    #[test]
    fn high_epsilon_contrast_stabilises_dipole() {
        // ε_in < ε_out ⇒ U_solv < 0.
        let u = dipole_solvation_energy(1.0, 10.0, 2.0, 80.0);
        assert!(u < 0.0);
    }
}
