//! Dielectric parameters and output unit conversions.
//!
//! Internal reduced-electrostatic convention: distances in Å, charges in
//! elementary charge `e`, energies in `e²/Å`. The solver never sees ε₀ or
//! `k_e` (only 4π inside the Green's function); those are output-layer
//! concerns and live here.

/// Coulomb constant in kJ·Å·mol⁻¹·e⁻².
///
/// Derived from CODATA: `k_e = 1/(4πε₀) · e² · N_A / (1000 J/kJ · 10⁻¹⁰ m/Å)`.
/// Numerically equal to `332.0637 × 4.184` (the kcal-based constant scaled
/// by the thermochemical calorie).
pub const KE_KJ_PER_MOL: f64 = 1389.3545708;

/// Ideal gas constant in kJ·mol⁻¹·K⁻¹.
pub const R_KJ_PER_MOL_PER_K: f64 = 8.314_462_618e-3;

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

/// Which side of the dielectric boundary the point charges live on.
///
/// `Interior` is the production case for MC pKa titration — charges
/// embedded in the low-ε protein interior Ω⁻. `Exterior` is the
/// validation-friendly case where charges sit in the high-ε solvent Ω⁺
/// (and the solver faces the salt/Yukawa regime directly). Placing the
/// Coulomb source term on the correct side of the Juffer block system
/// follows from this choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChargeSide {
    Interior,
    Exterior,
}

impl Dielectric {
    /// Pure dielectric (κ = 0). Shorthand for `continuum_with_salt(..., 0.0)`.
    #[must_use]
    pub const fn continuum(eps_in: f64, eps_out: f64) -> Self {
        Self::continuum_with_salt(eps_in, eps_out, 0.0)
    }

    /// Dielectric with salt (nonzero inverse Debye length).
    #[must_use]
    pub const fn continuum_with_salt(eps_in: f64, eps_out: f64, kappa: f64) -> Self {
        Self {
            eps_in,
            eps_out,
            kappa,
        }
    }
}

/// Convert a reduced-units energy (e²/Å) to kJ/mol.
#[allow(non_snake_case)]
pub const fn to_kJ_per_mol(energy_reduced: f64) -> f64 {
    energy_reduced * KE_KJ_PER_MOL
}

/// Convert a reduced-units energy (e²/Å) to dimensionless `kT` at `temperature_k`.
pub const fn to_kt(energy_reduced: f64, temperature_k: f64) -> f64 {
    to_kJ_per_mol(energy_reduced) / (R_KJ_PER_MOL_PER_K * temperature_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coulomb_at_1_angstrom_matches_literature() {
        // Two unit charges 1 Å apart in vacuum: U = 1 e²/Å = 1389.3546 kJ/mol.
        assert!((to_kJ_per_mol(1.0) - 1389.3545708).abs() < 1e-3);
    }

    #[test]
    fn kt_at_298k_is_2_479_kj() {
        // R·T at 298.15 K ≈ 2.479 kJ/mol — textbook value.
        let kt_per_kj = R_KJ_PER_MOL_PER_K * 298.15;
        assert!((kt_per_kj - 2.479).abs() < 1e-3);
    }
}
