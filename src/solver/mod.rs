//! BEM solve + field evaluation.
//!
//! The only public items are [`BemSolution`] and its inherent methods.
//! Panel integrals, block-matrix layout, faer types, and `DVec3` stay
//! crate-internal.

mod assembly;
mod linalg;
mod panel_integrals;

use crate::error::{Error, Result};
use crate::geometry::Surface;
use crate::units::Dielectric;
use glam::DVec3;

/// Solved surface densities for a given [`Surface`], [`Dielectric`] and
/// charge configuration. Immutable after construction.
///
/// Holds the surface potential `f` and its outward normal derivative `h` at
/// each face centroid. All field-evaluation methods read these densities.
pub struct BemSolution<'s> {
    surface: &'s Surface,
    media: Dielectric,
    /// Surface potential at each face centroid (reduced units, e/Å).
    f: Vec<f64>,
    /// ∂φ_out/∂n at each face centroid (reduced units, e/Å²).
    h: Vec<f64>,
}

impl<'s> BemSolution<'s> {
    /// Solve the Juffer BIE for the given surface, dielectric, and charges.
    ///
    /// `charge_positions` and `charge_values` must have the same length.
    /// Positions are in Å; values are in elementary charge units `e`.
    ///
    /// # Errors
    /// Returns [`Error::ChargeLenMismatch`] if the input arrays have
    /// different lengths, or [`Error::SolveFailed`] if the LU produces a
    /// non-finite solution (ill-conditioned system).
    pub fn solve(
        surface: &'s Surface,
        media: Dielectric,
        charge_positions: &[[f64; 3]],
        charge_values: &[f64],
    ) -> Result<Self> {
        if charge_positions.len() != charge_values.len() {
            return Err(Error::ChargeLenMismatch {
                positions: charge_positions.len(),
                values: charge_values.len(),
            });
        }
        // why: κ > 0 requires a second block with Yukawa kernels alongside
        // K₀, K₀'; deferred to the next milestone.
        assert!(media.kappa == 0.0, "κ > 0 not yet implemented");

        let (a_matrix, rhs) =
            assembly::build_block_system(surface, media, charge_positions, charge_values);
        let solution = linalg::solve_dense(a_matrix, rhs)?;

        let n = surface.num_faces();
        // why: block ordering in assembly is [f; h] (top = potential,
        // bottom = normal derivative) — see assembly::build_block_system.
        let (f, h) = solution.split_at(n);
        Ok(Self {
            surface,
            media,
            f: f.to_vec(),
            h: h.to_vec(),
        })
    }

    /// Surface potential `f` at each face centroid (reduced units).
    pub fn surface_potential(&self) -> &[f64] {
        &self.f
    }

    /// Normal derivative `h = ∂φ_out/∂n` at each face centroid (reduced units).
    pub fn surface_normal_deriv(&self) -> &[f64] {
        &self.h
    }

    /// Reaction-field potential at a single external point (reduced units, e/Å).
    ///
    /// Evaluates the exterior Green's-3rd-identity representation
    ///   `φ_rf(r) = ∫_Γ [ f · ∂_{n,s}G₀(r, s') − G₀(r, s') · h ] dS'`
    /// using a 3-point Gauss rule per panel. The Coulomb contribution is
    /// *not* included — this is φ − φ_coul.
    pub fn reaction_field_at(&self, point: [f64; 3]) -> f64 {
        reaction_field_at_impl(self.surface, &self.f, &self.h, DVec3::from(point))
    }

    /// Batched reaction-field evaluation at many external points.
    ///
    /// `out` is filled with `φ_rf` at each point. `out.len()` must equal
    /// `points.len()`; otherwise [`Error::OutputBufferTooSmall`].
    pub fn reaction_field_at_many(&self, points: &[[f64; 3]], out: &mut [f64]) -> Result<()> {
        if out.len() < points.len() {
            return Err(Error::OutputBufferTooSmall {
                expected: points.len(),
                got: out.len(),
            });
        }
        for (p, o) in points.iter().zip(out.iter_mut()) {
            *o = self.reaction_field_at(*p);
        }
        Ok(())
    }

    /// Reaction-field contribution to the pairwise interaction energy
    /// `W_ij` between charge `i` and charge `j` (reduced units, e²/Å).
    ///
    /// Equals `q_j · φ_rf(r_j)` when the solve was seeded with the source
    /// charge `q_i` at `r_i`. The caller passes the same arrays that were
    /// used in [`Self::solve`] so we can index `i` and `j` consistently.
    pub fn interaction_energy(
        &self,
        charge_positions: &[[f64; 3]],
        charge_values: &[f64],
        _i: usize,
        j: usize,
    ) -> Result<f64> {
        if charge_positions.len() != charge_values.len() {
            return Err(Error::ChargeLenMismatch {
                positions: charge_positions.len(),
                values: charge_values.len(),
            });
        }
        if j >= charge_values.len() {
            return Err(Error::ChargeIndexOutOfRange {
                index: j,
                count: charge_values.len(),
            });
        }
        // why: for a solve seeded with *all* source charges, φ_rf at r_j
        // already encodes the pairwise cross-terms plus Born self-energy
        // of j. The caller subtracts diagonals as needed downstream.
        let phi = self.reaction_field_at(charge_positions[j]);
        Ok(charge_values[j] * phi)
    }

    /// Dielectric used for the solve (passthrough for convenience).
    pub fn dielectric(&self) -> Dielectric {
        self.media
    }
}

fn reaction_field_at_impl(surface: &Surface, f: &[f64], h: &[f64], r: DVec3) -> f64 {
    // why: evaluator uses the same 3-point Gauss rule as the assembly
    // off-diagonal so the two stages share convergence order. Sign follows
    // Green's 3rd identity applied to Ω⁺ (outward normal −n):
    //   φ_rf = ∫_Γ [f · ∂_{n,s}G − G · h] dS'.
    let geom = surface.geom_internal();
    let mut sum = 0.0;
    for b in 0..geom.len() {
        let nb = geom.normals[b];
        let ab = geom.areas[b];
        let mut quad = 0.0;
        for p in panel_integrals::gauss3_points(geom.tris[b]) {
            let d = r - p;
            let inv_r = 1.0 / d.length();
            let g0 = inv_r / (4.0 * core::f64::consts::PI);
            let dg0_dn_source = d.dot(nb) * inv_r * inv_r * inv_r / (4.0 * core::f64::consts::PI);
            quad += f[b] * dg0_dn_source - g0 * h[b];
        }
        sum += quad * ab / 3.0;
    }
    sum
}
