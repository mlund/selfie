//! Build the RHS of the Juffer derivative-BIE block system.
//!
//! ```text
//! [ ½I + K₀'     −(ε_out/ε_in) K₀ ] [ f ]   [ φ_coul_in  ]
//! [ ½I − K_κ'            K_κ      ] [ h ] = [ φ_yukawa   ]
//! ```
//!
//! Only one of the two RHS blocks is nonzero, determined by
//! [`crate::units::ChargeSide`]:
//! - **Interior sources** (protein-internal ionizables): top RHS carries
//!   the Laplace Coulomb `φ_coul_in(c_a) = Σ q_i / (ε_in · |c_a − r_i|)`;
//!   bottom RHS is zero.
//! - **Exterior sources** (validation convention): bottom RHS carries the
//!   screened Coulomb `φ_yukawa(c_a) = Σ q_i · exp(−κ|c_a − r_i|) /
//!   (ε_out · |c_a − r_i|)`; top RHS is zero.
//!
//! The block matrix itself is applied matrix-free by
//! [`crate::solver::operator::BemOperator`].
//!
//! Reference: Juffer et al., *J. Comput. Phys.* 97 (1991) 144–171,
//! https://doi.org/10.1016/0021-9991(91)90043-K.

use crate::geometry::Surface;
use crate::units::{ChargeSide, Dielectric};
use glam::DVec3;

pub(super) fn build_rhs(
    surface: &Surface,
    media: Dielectric,
    side: ChargeSide,
    charge_positions: &[[f64; 3]],
    charge_values: &[f64],
) -> Vec<f64> {
    let geom = surface.geom_internal();
    let n = geom.len();
    let mut rhs = vec![0.0_f64; 2 * n];

    let (offset, eps_source, kappa_source) = match side {
        // Interior is never screened — κ applies only to the exterior
        // solvent. Top block receives the Laplace Coulomb potential.
        ChargeSide::Interior => (0, media.eps_in, 0.0),
        // Exterior sources: bottom block, screened Coulomb at the
        // user-specified ionic strength.
        ChargeSide::Exterior => (n, media.eps_out, media.kappa),
    };

    // why: Interior RHS is always κ=0 (interior never screens). Split
    // the Coulomb vs Yukawa branch out of the per-(a, q) inner loop
    // so ~M×N exp() evaluations vanish whenever sources are interior
    // or salt-free.
    if kappa_source == 0.0 {
        for (a, out) in rhs.iter_mut().skip(offset).take(n).enumerate() {
            let ca = geom.centroids[a];
            let mut phi = 0.0;
            for (pos, &q) in charge_positions.iter().zip(charge_values.iter()) {
                let dist = (ca - DVec3::from(*pos)).length();
                phi += q / (eps_source * dist);
            }
            *out = phi;
        }
    } else {
        for (a, out) in rhs.iter_mut().skip(offset).take(n).enumerate() {
            let ca = geom.centroids[a];
            let mut phi = 0.0;
            for (pos, &q) in charge_positions.iter().zip(charge_values.iter()) {
                let dist = (ca - DVec3::from(*pos)).length();
                phi += q * crate::solver::panel_integrals::exp_neg(kappa_source * dist)
                    / (eps_source * dist);
            }
            *out = phi;
        }
    }

    rhs
}
