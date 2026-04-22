//! Dielectric parameters and output unit conversions.
//!
//! Internal reduced-electrostatic convention: distances in Å, charges in
//! elementary charge `e`, energies in `e²/Å`. The solver never sees ε₀ or
//! `k_e` (only 4π inside the Green's function); those are output-layer
//! concerns and live here.

/// Coulomb constant in kcal·Å·mol⁻¹·e⁻².
///
/// Derived from CODATA: `k_e = 1/(4πε₀) · e² · N_A / (4184 J/kcal · 10⁻¹⁰ m/Å)`.
pub const KE_KCAL_PER_MOL: f64 = 332.0637;

/// Ideal gas constant in kcal·mol⁻¹·K⁻¹.
pub const R_KCAL_PER_MOL_PER_K: f64 = 1.987_204_259e-3;

/// Continuum-dielectric description of the interior and exterior media.
///
/// `kappa` is the inverse Debye length for the exterior Poisson–Boltzmann
/// solution, in Å⁻¹. Phase 1 targets `kappa = 0` (pure dielectric).
#[derive(Clone, Copy, Debug)]
pub struct Dielectric {
    pub eps_in: f64,
    pub eps_out: f64,
    pub kappa: f64,
}

impl Dielectric {
    /// Pure dielectric (κ = 0).
    pub fn continuum(eps_in: f64, eps_out: f64) -> Self {
        Self {
            eps_in,
            eps_out,
            kappa: 0.0,
        }
    }

    /// Dielectric with salt (nonzero inverse Debye length).
    pub fn continuum_with_salt(eps_in: f64, eps_out: f64, kappa: f64) -> Self {
        Self {
            eps_in,
            eps_out,
            kappa,
        }
    }
}

/// Convert a reduced-units energy (e²/Å) to kcal/mol.
pub fn to_kcal_per_mol(energy_reduced: f64) -> f64 {
    energy_reduced * KE_KCAL_PER_MOL
}

/// Convert a reduced-units energy (e²/Å) to dimensionless `kT` at `temperature_k`.
pub fn to_kt(energy_reduced: f64, temperature_k: f64) -> f64 {
    to_kcal_per_mol(energy_reduced) / (R_KCAL_PER_MOL_PER_K * temperature_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coulomb_at_1_angstrom_matches_literature() {
        // Two unit charges 1 Å apart in vacuum: U = +1 e²/Å = 332.0637 kcal/mol.
        assert!((to_kcal_per_mol(1.0) - 332.0637).abs() < 1e-3);
    }

    #[test]
    fn kt_at_298k_is_0_5921_kcal() {
        // R·T at 298 K ≈ 0.5921 kcal/mol — textbook value.
        let kt_per_kcal = R_KCAL_PER_MOL_PER_K * 298.15;
        assert!((kt_per_kcal - 0.5924).abs() < 1e-3);
    }
}
