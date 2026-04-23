//! Real-protein tests on pygbe's lysozyme single-surface setup.
//!
//! pygbe's `tests/convergence_tests/input_files/lys_single_1.config`
//! solves lysozyme with a single closed protein surface (no Stern layer,
//! no internal cavities) against bulk water at κ = 0.125 Å⁻¹ — exactly
//! our physics setup. The Lys1 mesh has 14,398 faces (28,796 unknowns).
//!
//! Two tests here:
//!
//! - [`lysozyme_single_surface_pipeline`]: fast I/O + geometry-sanity
//!   pipeline on a realistic workload. Runs by default under
//!   `--features validation`.
//! - [`lysozyme_full_solve`]: end-to-end GMRES solve. `#[ignore]`'d so
//!   it stays out of default `cargo test`; run with
//!   `cargo test --release --features validation -- --ignored`.

#![cfg(feature = "validation")]

use selfie::io::{Charges, read_msms, read_pqr};
use selfie::units::to_kcal_per_mol;
use selfie::{BemSolution, ChargeSide, Dielectric, Surface};
use std::path::PathBuf;

fn fixture(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pygbe_lys")
        .join(sub)
}

fn load_lysozyme() -> (Surface, Charges) {
    let (vertices, faces) =
        read_msms(fixture("Lys1.vert"), fixture("Lys1.face")).expect("read_msms");
    let surface = Surface::from_mesh(&vertices, &faces)
        .expect("Lys1 mesh should be a closed orientable 2-manifold");
    let charges = read_pqr(fixture("built_parse.pqr")).expect("read_pqr");
    (surface, charges)
}

#[test]
fn lysozyme_single_surface_pipeline() {
    let (surface, charges) = load_lysozyme();
    // MSMS dedup: Lys1 ships shared-vertex (not triangle-soup), so the
    // raw counts already reflect the topological structure.
    assert_eq!(surface.num_faces(), 14_398);
    assert_eq!(surface.num_vertices(), 7_201);
    assert_eq!(charges.values.len(), 1_323);

    // All atoms are inside the SES, by construction.
    let side = surface
        .classify_charges(&charges.positions)
        .expect("classify_charges must produce a single side");
    assert_eq!(side, ChargeSide::Interior);

    // Sanity on the total integer charge: protein at neutral pH should
    // be within a few units of zero.
    let total: f64 = charges.values.iter().sum();
    assert!(
        total.abs() < 20.0,
        "unreasonable total charge: {total}"
    );
    eprintln!(
        "lys_single_1 fixture: {} faces, {} atoms, total charge {:+.3} e",
        surface.num_faces(),
        charges.values.len(),
        total
    );
}

/// End-to-end GMRES solve on the full 14,398-face lysozyme mesh.
///
/// `#[ignore]`'d because even with the RAS preconditioner each solve
/// still costs ~20 s of CPU — fine for opt-in regression, too slow for
/// every `cargo test` invocation. Run explicitly with
/// `cargo test --release --features validation --test pygbe_lys_mesh -- --ignored`.
///
/// Reference: pygbe's `lys_single_1` convergence test
/// (`tests/convergence_tests/lysozyme.py`, `Esolv_ref_single[0]`), which
/// fixes `E_solv = −2401.2 kJ/mol ≈ −573.90 kcal/mol` at the Lys1 mesh
/// with ε_in=4, ε_out=80, κ=0.125 Å⁻¹. We use the same mesh
/// (`Lys1.{vert,face}`) and the same charge file (`built_parse.pqr`),
/// and with the neighbour-block preconditioner our E_solv matches
/// pygbe's to ≈ 0.1 %.
#[test]
#[ignore]
fn lysozyme_full_solve() {
    const PYGBE_ESOLV_SINGLE_LYS1_KCAL: f64 = -2401.2 / 4.184;
    const REL_TOL: f64 = 0.05;

    let (surface, charges) = load_lysozyme();
    let media = Dielectric::continuum_with_salt(4.0, 80.0, 0.125);

    let t = std::time::Instant::now();
    let solution = BemSolution::solve(
        &surface,
        media,
        ChargeSide::Interior,
        &charges.positions,
        &charges.values,
    )
    .expect("GMRES should converge on a well-conditioned BIE");
    eprintln!(
        "lysozyme full solve: {} faces, {} atoms, elapsed {:.1}s",
        surface.num_faces(),
        charges.values.len(),
        t.elapsed().as_secs_f64()
    );

    // E_solv = ½ Σ qᵢ φ_rf(rᵢ), summed over all source charges.
    let e_solv_reduced: f64 = (0..charges.values.len())
        .map(|j| {
            solution
                .interaction_energy(&charges.positions, &charges.values, j, j)
                .expect("interaction_energy")
        })
        .sum::<f64>()
        * 0.5;
    let e_solv_kcal = to_kcal_per_mol(e_solv_reduced);
    let rel_err = (e_solv_kcal - PYGBE_ESOLV_SINGLE_LYS1_KCAL).abs()
        / PYGBE_ESOLV_SINGLE_LYS1_KCAL.abs();
    eprintln!(
        "E_solv (Lys1 single-surface): ours = {:+.2} kcal/mol, \
         pygbe = {:+.2} kcal/mol, rel. err = {:.2}%",
        e_solv_kcal,
        PYGBE_ESOLV_SINGLE_LYS1_KCAL,
        rel_err * 100.0,
    );
    assert!(
        rel_err < REL_TOL,
        "single-surface lysozyme disagrees with pygbe by {:.2}% (tol {:.0}%)",
        rel_err * 100.0,
        REL_TOL * 100.0,
    );
}
