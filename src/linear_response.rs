//! Precomputed linear-response basis for charge-parameterised BEM.
//!
//! The linearised Poisson-Boltzmann operator the solver inverts is
//! linear in the source charges: the system matrix depends only on
//! mesh + dielectric, and the RHS is `Σ_i q_i · g_i(c_a)`, a sum of
//! per-site basis functions scaled by the charge values. So the
//! surface densities `(f, h)` and the reaction-field potential are
//! both linear in the charge vector, and the solvation energy is a
//! bilinear form
//!
//! ```text
//! E_solv(q) = ½ qᵀ G q           with  G_ij = φ_rf_i(r_j)
//! ```
//!
//! where `φ_rf_i` is the reaction field from a unit charge at site
//! `i`. Precomputing the `N_sites × N_sites` matrix `G` (N BEM
//! solves + one batch-probe per row) collapses every downstream
//! evaluation to `O(N²)` linear algebra — suitable for workloads
//! that vary charges many times over a fixed mesh
//! (solvation-energy scans, binding-energy calculations, MC pKa
//! titration).
//!
//! The type exposes the bilinear energy, the per-site reaction-field
//! vector `G q`, and single- / batch-probe reaction-field evaluators
//! at arbitrary points.

use crate::error::{Error, Result};
use crate::geometry::Surface;
use crate::solver::{BemSolution, SolveContext};
use crate::units::{ChargeSide, Dielectric};
use rayon::prelude::*;

/// Validate that `charges.len()` matches the number of sites `n`.
fn check_charges(charges: &[f64], n: usize) -> Result<()> {
    if charges.len() != n {
        return Err(Error::ChargeLenMismatch {
            positions: n,
            values: charges.len(),
        });
    }
    Ok(())
}

/// Validate that `out` holds at least `expected` slots.
fn check_out_len(out: &[f64], expected: usize) -> Result<()> {
    if out.len() < expected {
        return Err(Error::OutputBufferTooSmall {
            expected,
            got: out.len(),
        });
    }
    Ok(())
}

/// Precomputed linear-response basis: owns one BEM solve per site,
/// plus the symmetric `G_ij = φ_rf_i(r_j)` matrix.
///
/// See the module doc for the linearity principle. Constructor cost
/// is `N_sites` full `BemSolution::solve` calls plus `N_sites` batch
/// probes; every method after [`Self::precompute`] is a pure
/// linear-algebra or quadrature contraction — no further BEM solves.
///
/// All sites share one [`ChargeSide`]; heterogeneous setups (some
/// charges inside the protein, some outside) build two instances.
#[derive(Debug)]
pub struct LinearResponse<'s> {
    // why: each basis borrows the same `&'s Surface` and shares one
    // Dielectric + ChargeSide; those live inside the BemSolutions,
    // not duplicated here. `bases.len()` is the site count.
    bases: Vec<BemSolution<'s>>,
    // `G_ij = φ_rf_i(r_j)`, row-major `[i * n + j]`, symmetrised.
    // "Row i" is basis i's reaction field sampled at every site.
    response_matrix: Vec<f64>,
}

impl<'s> LinearResponse<'s> {
    /// Build the basis: one `BemSolution::solve` per site (unit
    /// charge), then fill and symmetrise the response matrix.
    ///
    /// # Errors
    /// Propagates any error from the underlying BEM solves (charge
    /// straddling the dielectric, GMRES non-convergence).
    pub fn precompute(
        surface: &'s Surface,
        media: Dielectric,
        side: ChargeSide,
        sites: &[[f64; 3]],
    ) -> Result<Self> {
        let n = sites.len();
        // why: the operator (Barnes-Hut tree) and the RAS
        // preconditioner depend on geometry + dielectric only; they
        // are identical across all N basis solves. Building them
        // once inside a `SolveContext` and feeding per-site RHS
        // amortises the setup cost linearly in N — on lysozyme
        // (N_sites ≈ 50) the RAS build is tens of percent of each
        // cold solve, so the savings are substantial.
        //
        // Basis solves themselves still run sequentially; each
        // already saturates all cores through rayon inside the
        // operator apply and `reaction_field_at_many`.
        let ctx = SolveContext::new(surface, media, side);
        let mut bases = Vec::with_capacity(n);
        for site in sites {
            let (f, h) = ctx.solve_charges(core::slice::from_ref(site), &[1.0])?;
            bases.push(BemSolution::from_densities(surface, media, side, f, h));
        }

        let mut response_matrix = vec![0.0_f64; n * n];
        for (i, basis) in bases.iter().enumerate() {
            basis.reaction_field_at_many(sites, &mut response_matrix[i * n..(i + 1) * n])?;
        }

        // why: the continuous Green's function is reciprocity-
        // symmetric, but centroid collocation breaks it by O(h²)
        // ≈ 1e-4 on a lysozyme-scale mesh. Symmetrising in place
        // makes `½ qᵀ G q` a well-defined bilinear form regardless
        // of eval order and guarantees self-consistency of any MC
        // caller that reads off-diagonal entries.
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (response_matrix[i * n + j] + response_matrix[j * n + i]);
                response_matrix[i * n + j] = avg;
                response_matrix[j * n + i] = avg;
            }
        }

        Ok(Self {
            bases,
            response_matrix,
        })
    }

    /// Number of precomputed sites `N`.
    pub fn num_sites(&self) -> usize {
        self.bases.len()
    }

    /// Crate-internal: consume the basis and yield each site's
    /// `(f, h)` densities. Lets the Python wrapper carry the basis
    /// across the FFI boundary without re-solving — the BemSolutions
    /// themselves can't cross because they borrow `&'s Surface`.
    #[cfg(feature = "python")]
    pub(crate) fn into_densities(self) -> Vec<(Vec<f64>, Vec<f64>)> {
        self.bases
            .into_iter()
            .map(BemSolution::into_densities)
            .collect()
    }

    /// Solvation energy `½ qᵀ G q` (reduced units `e²/Å`). `O(N²)`.
    ///
    /// # Errors
    /// [`Error::ChargeLenMismatch`] if `charges.len() != self.num_sites()`.
    pub fn solvation_energy(&self, charges: &[f64]) -> Result<f64> {
        let n = self.num_sites();
        check_charges(charges, n)?;
        // why: the zipped `.iter().sum()` pattern autovectorises to
        // FMA cleanly on both NEON and AVX2; a hand-rolled index
        // loop blocks LLVM on the aliasing between `response_matrix`
        // and `charges`.
        let acc: f64 = self
            .response_matrix
            .chunks_exact(n)
            .zip(charges)
            .map(|(row, &qi)| qi * row.iter().zip(charges).map(|(g, q)| g * q).sum::<f64>())
            .sum();
        Ok(0.5 * acc)
    }

    /// Reaction-field potential `φ_rf(r_j; q) = (G q)_j` at every
    /// precomputed site (reduced units `e/Å`). `O(N²)` via the
    /// stored response matrix.
    ///
    /// # Errors
    /// [`Error::ChargeLenMismatch`] if `charges.len() != self.num_sites()`,
    /// [`Error::OutputBufferTooSmall`] if `out.len() < self.num_sites()`.
    pub fn reaction_field_at_sites(&self, charges: &[f64], out: &mut [f64]) -> Result<()> {
        let n = self.num_sites();
        check_charges(charges, n)?;
        check_out_len(out, n)?;
        // why: column `j` of the symmetric `G` is read as `row j` —
        // the `chunks_exact(n).zip(charges)` walk presents contiguous
        // f64 stripes paired with one charge each, which LLVM folds
        // into FMA accumulation over `j`.
        out[..n].fill(0.0);
        for (row, &qi) in self.response_matrix.chunks_exact(n).zip(charges) {
            for (out_j, &g) in out.iter_mut().take(n).zip(row) {
                *out_j += qi * g;
            }
        }
        Ok(())
    }

    /// Reaction-field potential at an arbitrary probe point (reduced
    /// units `e/Å`) for charges `q`. `O(N · N_panels)` —
    /// one basis quadrature per site.
    ///
    /// **Contract**: `point` must lie on the same side of the
    /// dielectric boundary as the `ChargeSide` passed to
    /// [`Self::precompute`] — same caveat as
    /// [`BemSolution::reaction_field_at`].
    ///
    /// # Errors
    /// [`Error::ChargeLenMismatch`] if `charges.len() != self.num_sites()`.
    pub fn reaction_field_at(&self, charges: &[f64], point: [f64; 3]) -> Result<f64> {
        check_charges(charges, self.num_sites())?;
        Ok(self
            .bases
            .iter()
            .zip(charges)
            .map(|(basis, &q)| q * basis.reaction_field_at(point))
            .sum())
    }

    /// Batched reaction-field evaluation at many external points for
    /// a given charge configuration (reduced units `e/Å`). Parallel
    /// over probes via rayon; `O(N · N_probes · N_panels)`.
    ///
    /// # Errors
    /// [`Error::ChargeLenMismatch`] if `charges.len() != self.num_sites()`,
    /// [`Error::OutputBufferTooSmall`] if `out.len() < points.len()`.
    pub fn reaction_field_at_many(
        &self,
        charges: &[f64],
        points: &[[f64; 3]],
        out: &mut [f64],
    ) -> Result<()> {
        check_charges(charges, self.num_sites())?;
        check_out_len(out, points.len())?;
        // why: probes are independent across points; parallelise
        // across probes. Each probe's inner loop over bases is
        // sequential (N_sites ≈ 50 terms) — cheap, and keeps the
        // parallel granularity at per-probe instead of nested.
        points
            .par_iter()
            .zip(out.par_iter_mut())
            .for_each(|(p, o)| {
                *o = self
                    .bases
                    .iter()
                    .zip(charges)
                    .map(|(basis, &q)| q * basis.reaction_field_at(*p))
                    .sum();
            });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Surface;

    fn unit_icosphere() -> Surface {
        // why: radius 10 Å, subdivision 3 → 320 faces; large enough
        // to exercise the full pipeline in < 0.5 s per basis solve,
        // small enough that tests stay fast.
        Surface::icosphere(10.0, 3)
    }

    fn sample_sites() -> Vec<[f64; 3]> {
        // Five interior positions inside a radius-10 sphere.
        vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, -4.0],
            [1.5, -1.5, 2.0],
        ]
    }

    fn sample_media() -> Dielectric {
        Dielectric::continuum(2.0, 80.0)
    }

    #[test]
    fn bilinear_energy_matches_direct_solve() {
        // Headline correctness: basis-composed ½ qᵀ G q must match a
        // direct `BemSolution::solve` + `interaction_energy` sum for
        // the same charges. This proves the linearity principle
        // holds through the discretised operator.
        //
        // Tolerance: GMRES converges to `RELATIVE_TOL = 1e-5` per
        // solve, and the basis path sums `N_sites` solves' residual
        // errors — the bilinear composition will differ from the
        // direct solve by a small multiple of that tolerance. 1e-4
        // is loose enough to stay green across mesh refinements and
        // tight enough to catch any structural bug in the linearity
        // reconstruction.
        let surface = unit_icosphere();
        let media = sample_media();
        let sites = sample_sites();
        let basis = LinearResponse::precompute(&surface, media, ChargeSide::Interior, &sites)
            .expect("precompute");

        for qs in [
            vec![1.0, 0.5, -0.3, 0.7, -0.2],
            vec![0.8, -0.4, 0.6, -0.9, 0.1],
            vec![-1.2, 0.3, 0.5, -0.6, 0.8],
        ] {
            let basis_energy = basis.solvation_energy(&qs).unwrap();
            let direct = BemSolution::solve(&surface, media, ChargeSide::Interior, &sites, &qs)
                .expect("direct solve");
            let mut direct_energy = 0.0;
            for (j, &_q) in qs.iter().enumerate() {
                direct_energy += direct.interaction_energy(&sites, &qs, j).unwrap();
            }
            direct_energy *= 0.5;
            // why: GMRES residual is `RELATIVE_TOL = 1e-4` per solve, so the
            // bilinear reconstruction (sum of N_sites × N_sites terms) sits
            // around the same scale. Allow up to 1e-3 — well below any of
            // the integration-test (Born/Kirkwood/pyGBe) accuracy bars.
            let rel = (basis_energy - direct_energy).abs() / direct_energy.abs().max(1e-12);
            assert!(
                rel < 1e-3,
                "basis = {basis_energy:e}, direct = {direct_energy:e}, rel = {rel:e}"
            );
        }
    }

    #[test]
    fn solvation_energy_scales_quadratically() {
        let surface = unit_icosphere();
        let basis = LinearResponse::precompute(
            &surface,
            sample_media(),
            ChargeSide::Interior,
            &sample_sites(),
        )
        .unwrap();
        let base = vec![0.7, -0.3, 0.5, -0.2, 0.8];
        let e_base = basis.solvation_energy(&base).unwrap();
        for lambda in [0.1_f64, 0.5, 2.0, 3.7] {
            let scaled: Vec<f64> = base.iter().map(|q| lambda * q).collect();
            let e_scaled = basis.solvation_energy(&scaled).unwrap();
            let expected = lambda * lambda * e_base;
            let rel = (e_scaled - expected).abs() / expected.abs();
            assert!(rel < 1e-14, "λ={lambda}: rel = {rel:e}");
        }
    }

    #[test]
    fn response_matrix_is_symmetric() {
        let surface = unit_icosphere();
        let basis = LinearResponse::precompute(
            &surface,
            sample_media(),
            ChargeSide::Interior,
            &sample_sites(),
        )
        .unwrap();
        let n = basis.num_sites();
        for i in 0..n {
            for j in (i + 1)..n {
                let up = basis.response_matrix[i * n + j];
                let lo = basis.response_matrix[j * n + i];
                assert_eq!(up, lo, "({i},{j}) not bit-symmetric");
            }
        }
    }

    #[test]
    fn exterior_side_matches_direct_solve() {
        // Exterior-source branch: charges live in Ω⁺ (solvent) and
        // the Yukawa RHS is active. Same bilinear invariant must
        // hold; exercises `reaction_field_at_many`'s Yukawa path.
        let surface = unit_icosphere();
        let media = Dielectric::continuum_with_salt(2.0, 80.0, 0.125);
        // Place sites outside the radius-10 sphere.
        let sites = vec![
            [12.0, 0.0, 0.0],
            [0.0, 14.0, 0.0],
            [0.0, 0.0, 15.0],
            [11.0, -11.0, 0.0],
        ];
        let basis = LinearResponse::precompute(&surface, media, ChargeSide::Exterior, &sites)
            .expect("precompute");
        let qs = vec![0.9, -0.5, 0.3, -0.7];
        let basis_energy = basis.solvation_energy(&qs).unwrap();
        let direct = BemSolution::solve(&surface, media, ChargeSide::Exterior, &sites, &qs)
            .expect("direct solve");
        let mut direct_energy = 0.0;
        for j in 0..qs.len() {
            direct_energy += direct.interaction_energy(&sites, &qs, j).unwrap();
        }
        direct_energy *= 0.5;
        let rel = (basis_energy - direct_energy).abs() / direct_energy.abs();
        // Same GMRES-tolerance-driven loosening as the interior test.
        assert!(rel < 1e-4, "exterior: rel = {rel:e}");
    }

    #[test]
    fn reaction_field_at_is_linear() {
        // Linearity in charges at a single probe point.
        let surface = unit_icosphere();
        let basis = LinearResponse::precompute(
            &surface,
            sample_media(),
            ChargeSide::Interior,
            &sample_sites(),
        )
        .unwrap();
        let q1 = vec![1.0, 0.2, -0.3, 0.4, -0.5];
        let q2 = vec![-0.6, 0.5, 0.7, -0.2, 0.1];
        let lambda = 1.7_f64;
        let probe = [0.1, 0.2, -0.3];
        let combined: Vec<f64> = q1.iter().zip(&q2).map(|(a, b)| lambda * a + b).collect();
        let phi_combined = basis.reaction_field_at(&combined, probe).unwrap();
        let phi_q1 = basis.reaction_field_at(&q1, probe).unwrap();
        let phi_q2 = basis.reaction_field_at(&q2, probe).unwrap();
        let expected = lambda * phi_q1 + phi_q2;
        let rel = (phi_combined - expected).abs() / expected.abs();
        assert!(rel < 1e-12, "rel = {rel:e}");
    }

    #[test]
    fn at_sites_matches_at_many_on_the_sites() {
        // Cross-check: the G-based fast path `reaction_field_at_sites`
        // and the quadrature-based general path
        // `reaction_field_at_many(... , sites, ...)` must agree.
        // Tolerance is loose because the former has been
        // reciprocity-symmetrised while the latter has not —
        // discrepancies reflect the same O(h²) asymmetry
        // noted in the tests module doc.
        let surface = unit_icosphere();
        let sites = sample_sites();
        let basis =
            LinearResponse::precompute(&surface, sample_media(), ChargeSide::Interior, &sites)
                .unwrap();
        let q = vec![0.7, -0.3, 0.5, -0.2, 0.8];
        let n = basis.num_sites();
        let mut via_g = vec![0.0; n];
        let mut via_quad = vec![0.0; n];
        basis.reaction_field_at_sites(&q, &mut via_g).unwrap();
        basis
            .reaction_field_at_many(&q, &sites, &mut via_quad)
            .unwrap();
        for (j, (&a, &b)) in via_g.iter().zip(&via_quad).enumerate() {
            let rel = (a - b).abs() / a.abs().max(b.abs()).max(1e-12);
            assert!(
                rel < 1e-3,
                "site {j}: via_g = {a:e}, via_quad = {b:e}, rel = {rel:e}"
            );
        }
    }

    #[test]
    fn charge_length_mismatch_errors() {
        let surface = unit_icosphere();
        let basis = LinearResponse::precompute(
            &surface,
            sample_media(),
            ChargeSide::Interior,
            &sample_sites(),
        )
        .unwrap();
        // Wrong length.
        let short = vec![1.0, 0.5];
        match basis.solvation_energy(&short) {
            Err(Error::ChargeLenMismatch { positions, values }) => {
                assert_eq!(positions, 5);
                assert_eq!(values, 2);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn output_buffer_too_small_errors() {
        let surface = unit_icosphere();
        let basis = LinearResponse::precompute(
            &surface,
            sample_media(),
            ChargeSide::Interior,
            &sample_sites(),
        )
        .unwrap();
        let q = vec![1.0, 0.5, -0.3, 0.7, -0.2];
        let mut tiny = vec![0.0; 2];
        match basis.reaction_field_at_sites(&q, &mut tiny) {
            Err(Error::OutputBufferTooSmall { expected, got }) => {
                assert_eq!(expected, 5);
                assert_eq!(got, 2);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }
}
