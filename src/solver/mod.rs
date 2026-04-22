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
use crate::units::{ChargeSide, Dielectric};
use glam::DVec3;
use rayon::prelude::*;

/// Solved surface densities for a given [`Surface`], [`Dielectric`] and
/// charge configuration. Immutable after construction.
///
/// Holds the surface potential `f` and its outward normal derivative `h` at
/// each face centroid. All field-evaluation methods read these densities.
#[derive(Debug)]
pub struct BemSolution<'s> {
    surface: &'s Surface,
    media: Dielectric,
    side: ChargeSide,
    /// Surface potential at each face centroid (reduced units, e/Å).
    f: Vec<f64>,
    /// ∂φ_out/∂n at each face centroid (reduced units, e/Å²).
    h: Vec<f64>,
}

impl<'s> BemSolution<'s> {
    /// Solve the Juffer BIE for the given surface, dielectric, and charges.
    ///
    /// `side` selects whether point charges live inside (Ω⁻, the protein
    /// interior — the production MC-titration case) or outside (Ω⁺, the
    /// solvent — used for validation against the outside-charge Kirkwood
    /// reference). Arrays must have matching lengths; positions in Å,
    /// values in elementary charge units `e`.
    ///
    /// Note: interior is always Laplace (κ affects only the exterior
    /// solvent). `media.kappa` is ignored by the interior-source RHS.
    ///
    /// # Errors
    /// Returns [`Error::ChargeLenMismatch`] if the input arrays have
    /// different lengths, or [`Error::SolveFailed`] if the LU produces a
    /// non-finite solution (ill-conditioned system).
    pub fn solve(
        surface: &'s Surface,
        media: Dielectric,
        side: ChargeSide,
        charge_positions: &[[f64; 3]],
        charge_values: &[f64],
    ) -> Result<Self> {
        if charge_positions.len() != charge_values.len() {
            return Err(Error::ChargeLenMismatch {
                positions: charge_positions.len(),
                values: charge_values.len(),
            });
        }
        let (a_matrix, rhs) =
            assembly::build_block_system(surface, media, side, charge_positions, charge_values);
        let mut f = linalg::solve_dense(a_matrix, rhs)?;

        // why: block ordering in assembly is [f; h] (top = potential,
        // bottom = normal derivative). split_off moves ownership of the
        // second half out in O(1) without a heap copy.
        let h = f.split_off(surface.num_faces());
        Ok(Self {
            surface,
            media,
            side,
            f,
            h,
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

    /// Reaction-field potential at a probe point (reduced units, e/Å).
    ///
    /// For interior solves the evaluator uses the Laplace Green's function
    /// `G_0` and Ω⁻-outward normal:
    ///   `φ_rf(r) = ∫_Γ [ (ε_out/ε_in)·G_0·h − f · ∂_{n,s}G_0 ] dS'`
    /// For exterior solves it uses the Yukawa kernel and the Ω⁺-outward
    /// normal sign:
    ///   `φ_rf(r) = ∫_Γ [ f · ∂_{n,s}G_κ − G_κ · h ] dS'`
    /// `κ = 0` recovers the plain Coulomb Green's function in the second
    /// form. The Coulomb / Yukawa source contribution is *not* included —
    /// this is φ − φ_source.
    ///
    /// **Contract**: `point` must lie on the same side of the dielectric
    /// boundary as the charges passed to [`Self::solve`]. Evaluating on
    /// the opposite side silently returns a wrong number — the BIE kernel
    /// and jump sign are side-specific. No runtime check is performed;
    /// callers that can't statically guarantee the side should inspect
    /// their mesh.
    pub fn reaction_field_at(&self, point: [f64; 3]) -> f64 {
        reaction_field_at_impl(
            self.surface,
            &self.f,
            &self.h,
            DVec3::from(point),
            self.side,
            self.media,
        )
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
        // why: each reaction_field_at call is O(n_panels) and independent;
        // parallel evaluation scales ~linearly with cores for typical
        // (many-points, large-mesh) use cases.
        points
            .par_iter()
            .zip(out.par_iter_mut())
            .for_each(|(p, o)| *o = self.reaction_field_at(*p));
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
    pub const fn dielectric(&self) -> Dielectric {
        self.media
    }
}

fn reaction_field_at_impl(
    surface: &Surface,
    f: &[f64],
    h: &[f64],
    r: DVec3,
    side: ChargeSide,
    media: Dielectric,
) -> f64 {
    // why: the sign in Green's 3rd identity flips between interior and
    // exterior because Ω⁺'s outward normal is −n while Ω⁻'s is +n. The
    // interior formula also carries the ε_out/ε_in factor that links
    // h = ∂φ_out/∂n to ∂φ_in/∂n through the dielectric BC. Interior is
    // always Laplace (κ applies only to the exterior solvent), so
    // interior evaluations use G_0 regardless of `media.kappa`.
    const FOUR_PI: f64 = 4.0 * core::f64::consts::PI;
    let geom = surface.geom_internal();
    let (kappa_eval, f_sign, h_coeff) = match side {
        ChargeSide::Interior => (0.0, -1.0, media.eps_out / media.eps_in),
        ChargeSide::Exterior => (media.kappa, 1.0, -1.0),
    };
    let panels = f
        .iter()
        .zip(h)
        .zip(&geom.normals)
        .zip(&geom.areas)
        .zip(&geom.tris);
    let mut sum = 0.0;
    for ((((&f_b, &h_b), &nb), &ab), &tri) in panels {
        let mut quad = 0.0;
        for p in panel_integrals::gauss3_points(tri) {
            let d = r - p;
            let dist = d.length();
            let inv_r = 1.0 / dist;
            let exp_kr = (-kappa_eval * dist).exp();
            let g = exp_kr * inv_r / FOUR_PI;
            let dg_dn_source =
                kappa_eval.mul_add(dist, 1.0) * exp_kr * d.dot(nb) * inv_r * inv_r * inv_r
                    / FOUR_PI;
            quad += (f_sign * f_b).mul_add(dg_dn_source, h_coeff * g * h_b);
        }
        sum += quad * ab / 3.0;
    }
    sum
}
