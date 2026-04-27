//! BEM solve + field evaluation.
//!
//! The only public items are [`BemSolution`] and its inherent methods.
//! Panel integrals, block-matrix layout, faer types, and `DVec3` stay
//! crate-internal.

mod assembly;
mod context;
mod gmres;
mod kernel;
mod operator;
mod panel_integrals;
mod precond;
mod treecode;

pub(crate) use context::SolveContext;

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
    /// different lengths, or [`Error::SolveFailed`] if GMRES fails to
    /// reach the convergence tolerance within its iteration budget or
    /// produces a non-finite solution.
    pub fn solve(
        surface: &'s Surface,
        media: Dielectric,
        side: ChargeSide,
        charge_positions: &[[f64; 3]],
        charge_values: &[f64],
    ) -> Result<Self> {
        let ctx = SolveContext::new(surface, media, side);
        let (f, h) = ctx.solve_charges(charge_positions, charge_values)?;
        Ok(Self {
            surface,
            media,
            side,
            f,
            h,
        })
    }

    /// Crate-internal constructor from pre-solved surface densities.
    /// `LinearResponse::precompute` uses this to build one
    /// `BemSolution` per basis from a shared [`SolveContext`],
    /// amortising the operator + preconditioner build across sites.
    pub(crate) const fn from_densities(
        surface: &'s Surface,
        media: Dielectric,
        side: ChargeSide,
        f: Vec<f64>,
        h: Vec<f64>,
    ) -> Self {
        Self {
            surface,
            media,
            side,
            f,
            h,
        }
    }

    /// Surface potential `f` at each face centroid (reduced units).
    pub fn surface_potential(&self) -> &[f64] {
        &self.f
    }

    /// Normal derivative `h = ∂φ_out/∂n` at each face centroid (reduced units).
    pub fn surface_normal_deriv(&self) -> &[f64] {
        &self.h
    }

    /// Crate-internal: consume the solution and move out the
    /// `(f, h)` density Vecs without copying. Used by the Python
    /// wrapper to materialise per-site densities from a
    /// [`LinearResponse`] without re-solving.
    #[cfg(feature = "python")]
    pub(crate) fn into_densities(self) -> (Vec<f64>, Vec<f64>) {
        (self.f, self.h)
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

    /// Sample `φ_rf` on a regular 3D grid and write it as an OpenDX
    /// (`.dx`) volume — the format APBS uses, loadable directly in
    /// PyMOL (`cmd.load("phi.dx"); cmd.isosurface(...)`) and VMD
    /// (`mol new phi.dx type dx`).
    ///
    /// `origin` is the (0, 0, 0) corner of the grid in Å; `spacing` is
    /// the per-axis step in Å (anisotropic grids are allowed); `dims`
    /// is the number of sample points along each axis. The total
    /// sample count is `dims[0] * dims[1] * dims[2]`.
    ///
    /// Output values are in reduced units (`e/Å`); rescale in the
    /// visualiser if you want kT/e or similar.
    ///
    /// # Errors
    /// [`Error::Io`] on filesystem failure.
    pub fn write_potential_dx(
        &self,
        path: impl AsRef<std::path::Path>,
        origin: [f64; 3],
        spacing: [f64; 3],
        dims: [usize; 3],
    ) -> Result<()> {
        let path = path.as_ref();
        let io_err = |e: std::io::Error| Error::Io {
            path: path.display().to_string(),
            reason: e.to_string(),
        };

        // why: APBS / OpenDX storage order is x outermost, z innermost
        // (index = k + nz*(j + ny*i)). Build the probe-point list in
        // the same order so we can write `values` in one pass.
        let [nx, ny, nz] = dims;
        let n_total = nx * ny * nz;
        let mut points: Vec<[f64; 3]> = Vec::with_capacity(n_total);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    points.push([
                        origin[0] + i as f64 * spacing[0],
                        origin[1] + j as f64 * spacing[1],
                        origin[2] + k as f64 * spacing[2],
                    ]);
                }
            }
        }
        let mut values = vec![0.0_f64; n_total];
        self.reaction_field_at_many(&points, &mut values)?;

        use std::io::Write;
        let file = std::fs::File::create(path).map_err(io_err)?;
        let mut w = std::io::BufWriter::new(file);
        writeln!(
            w,
            "# OpenDX scalar field — reaction-field potential φ_rf (reduced units, e/Å)"
        )
        .map_err(io_err)?;
        writeln!(w, "# Generated by BEMtzmann").map_err(io_err)?;
        writeln!(w, "object 1 class gridpositions counts {nx} {ny} {nz}").map_err(io_err)?;
        writeln!(w, "origin {} {} {}", origin[0], origin[1], origin[2]).map_err(io_err)?;
        writeln!(w, "delta {} 0 0", spacing[0]).map_err(io_err)?;
        writeln!(w, "delta 0 {} 0", spacing[1]).map_err(io_err)?;
        writeln!(w, "delta 0 0 {}", spacing[2]).map_err(io_err)?;
        writeln!(w, "object 2 class gridconnections counts {nx} {ny} {nz}").map_err(io_err)?;
        writeln!(
            w,
            "object 3 class array type double rank 0 items {n_total} data follows"
        )
        .map_err(io_err)?;
        // Three values per line keeps the file readable in a text editor
        // and matches the convention APBS uses.
        for chunk in values.chunks(3) {
            for (i, v) in chunk.iter().enumerate() {
                if i > 0 {
                    write!(w, " ").map_err(io_err)?;
                }
                write!(w, "{v:.7e}").map_err(io_err)?;
            }
            writeln!(w).map_err(io_err)?;
        }
        writeln!(w, "attribute \"dep\" string \"positions\"").map_err(io_err)?;
        writeln!(
            w,
            "object \"regular positions regular connections\" class field"
        )
        .map_err(io_err)?;
        writeln!(w, "component \"positions\" value 1").map_err(io_err)?;
        writeln!(w, "component \"connections\" value 2").map_err(io_err)?;
        writeln!(w, "component \"data\" value 3").map_err(io_err)?;
        w.flush().map_err(io_err)?;
        Ok(())
    }

    /// Reaction-field contribution to charge `j`'s self-interaction
    /// `q_j · φ_rf(r_j)` (reduced units, e²/Å), evaluated against the
    /// already-built solution.
    ///
    /// Note: `φ_rf` was computed once for *all* source charges, so the
    /// returned value already includes the cross-terms with every other
    /// charge — it is the contribution of site `j` to `2 · E_solv`, not
    /// the pairwise W_ij. The caller passes the same arrays used in
    /// [`Self::solve`] so positions stay aligned with values.
    pub fn interaction_energy(
        &self,
        charge_positions: &[[f64; 3]],
        charge_values: &[f64],
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
        let points: panel_integrals::GaussPoints<3> = tri.into();
        let mut quad = 0.0;
        for i in 0..3 {
            let dx = r.x - points.xs[i];
            let dy = r.y - points.ys[i];
            let dz = r.z - points.zs[i];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let inv_r = 1.0 / dist;
            let exp_kr = (-kappa_eval * dist).exp();
            let g = exp_kr * inv_r / FOUR_PI;
            let d_dot_nb = dx * nb.x + dy * nb.y + dz * nb.z;
            let dg_dn_source =
                kappa_eval.mul_add(dist, 1.0) * exp_kr * d_dot_nb * inv_r * inv_r * inv_r / FOUR_PI;
            quad += points.ws[i] * (f_sign * f_b).mul_add(dg_dn_source, h_coeff * g * h_b);
        }
        sum += quad * ab;
    }
    sum
}

#[cfg(test)]
mod write_potential_dx_tests {
    use super::*;

    #[test]
    fn round_trip_grid_structure() {
        let surface = Surface::icosphere(10.0, 3);
        let media = Dielectric::continuum(2.0, 80.0);
        let sol = BemSolution::solve(
            &surface,
            media,
            ChargeSide::Interior,
            &[[0.0, 0.0, 0.0]],
            &[1.0],
        )
        .unwrap();

        let path =
            std::env::temp_dir().join(format!("bemtzmann_dx_test_{}.dx", std::process::id()));
        sol.write_potential_dx(&path, [-2.0, -2.0, -2.0], [1.0, 1.0, 1.0], [4, 4, 4])
            .unwrap();

        let content = std::fs::read_to_string(&path).expect("read back");
        assert!(content.contains("object 1 class gridpositions counts 4 4 4"));
        assert!(content.contains("origin -2 -2 -2"));
        assert!(content.contains("delta 1 0 0"));
        assert!(content.contains("items 64 data follows"));
        assert!(content.contains("class field"));

        // Count actual scalar entries between the `data follows` line and
        // the trailing `attribute "dep"` line: must equal 4·4·4 = 64.
        let body = content
            .split("data follows\n")
            .nth(1)
            .unwrap()
            .split("attribute")
            .next()
            .unwrap();
        let n_values = body.split_whitespace().count();
        assert_eq!(n_values, 64);

        std::fs::remove_file(&path).ok();
    }
}
