//! Real-protein tests on pygbe's lysozyme single-surface setup.
//!
//! pygbe's `tests/convergence_tests/input_files/lys_single_1.config`
//! solves lysozyme with a single closed protein surface (no Stern layer,
//! no internal cavities) against bulk water at κ = 0.125 Å⁻¹ — exactly
//! our physics setup. The Lys1 mesh has 14,398 faces (28,796 unknowns).
//!
//! All tests here depend on the pygbe mesh archive (Zenodo record
//! 55349), which [`tests/common`] fetches on demand into a local
//! cache. Because of that network prerequisite all tests in this
//! file are `#[ignore]`'d; run with
//! `cargo test --release --features validation -- --ignored`.

#![cfg(feature = "validation")]

use selfie::io::{Charges, read_msms, read_pqr};
use selfie::units::to_kJ_per_mol;
use selfie::{BemSolution, ChargeSide, Dielectric, Surface};
use std::path::PathBuf;

mod common;

fn charge_fixture() -> PathBuf {
    // PQR stays in the repo — not part of the pygbe mesh archive.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pygbe_lys/built_parse.pqr")
}

fn load_lysozyme_at(resolution: &str) -> (Surface, Charges) {
    let dir = common::pygbe_lysozyme_dir();
    let vert = dir.join(format!("Lys{resolution}.vert"));
    let face = dir.join(format!("Lys{resolution}.face"));
    let (vertices, faces) = read_msms(&vert, &face).expect("read_msms");
    let surface = Surface::from_mesh(&vertices, &faces)
        .expect("lysozyme mesh should be a closed orientable 2-manifold");
    let charges = read_pqr(charge_fixture()).expect("read_pqr");
    (surface, charges)
}

fn load_lysozyme() -> (Surface, Charges) {
    load_lysozyme_at("1")
}

/// `#[ignore]`-gated because it now fetches the pygbe mesh archive
/// from Zenodo on first use (≈9 MB). Kept as a coarse sanity check
/// — no GMRES solve, just load + classify — for opt-in runs.
#[test]
#[ignore]
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
    assert!(total.abs() < 20.0, "unreasonable total charge: {total}");
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
    const PYGBE_ESOLV_SINGLE_LYS1_KJ: f64 = -2401.2;
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
    let e_solv_kj = to_kJ_per_mol(e_solv_reduced);
    let rel_err = (e_solv_kj - PYGBE_ESOLV_SINGLE_LYS1_KJ).abs() / PYGBE_ESOLV_SINGLE_LYS1_KJ.abs();
    eprintln!(
        "E_solv (Lys1 single-surface): ours = {:+.2} kJ/mol, \
         pygbe = {:+.2} kJ/mol, rel. err = {:.2}%",
        e_solv_kj, PYGBE_ESOLV_SINGLE_LYS1_KJ, rel_err * 100.0,
    );
    assert!(
        rel_err < REL_TOL,
        "single-surface lysozyme disagrees with pygbe by {:.2}% (tol {:.0}%)",
        rel_err * 100.0,
        REL_TOL * 100.0,
    );
}

fn solve_and_report(label: &str, surface: &Surface, charges: &Charges) -> f64 {
    let media = Dielectric::continuum_with_salt(4.0, 80.0, 0.125);
    let t = std::time::Instant::now();
    let sol = BemSolution::solve(
        surface,
        media,
        ChargeSide::Interior,
        &charges.positions,
        &charges.values,
    )
    .expect("GMRES should converge");
    let elapsed = t.elapsed().as_secs_f64();
    let e_solv_reduced: f64 = (0..charges.values.len())
        .map(|j| {
            sol.interaction_energy(&charges.positions, &charges.values, j, j)
                .expect("interaction_energy")
        })
        .sum::<f64>()
        * 0.5;
    let e_solv_kj = to_kJ_per_mol(e_solv_reduced);
    eprintln!(
        "{label}: {} faces, elapsed {elapsed:.1}s, E_solv = {e_solv_kj:+.2} kJ/mol",
        surface.num_faces()
    );
    e_solv_kj
}

/// Mesh-refinement gate on the real-protein path. Walks pygbe's
/// published `Lys{1,2,4,8}` single-surface convergence series and
/// checks our E_solv against their stored references (from
/// `pygbe/tests/convergence_tests/lysozyme.py`). `#[ignore]`-gated
/// because the Lys8 solve is ~25 s on a laptop.
///
/// pygbe reference values (`lys.param`, interior charges, ε_in = 4,
/// ε_out = 80, κ = 0.125 Å⁻¹):
///   Lys1 → −2401.2 kJ/mol  (14 398 faces)
///   Lys2 → −2161.8 kJ/mol  (~30 k faces)
///   Lys4 → −2089.0 kJ/mol  (~50 k faces)
///   Lys8 → −2065.5 kJ/mol  (~81 k faces)
///
/// The series converges from above — as the mesh refines the
/// surface approaches the SES limit and E_solv settles toward
/// ≈ −2050 kJ/mol. We check each resolution against pygbe to
/// ≤ 2 % (own MSMS artifacts + BEM order limit the absolute
/// match), and assert the monotone-convergence pattern.
#[test]
#[ignore]
fn lysozyme_mesh_refinement_matches_pygbe_series() {
    // Reference values from pygbe's convergence_tests/lysozyme.py
    // (`Esolv_ref_single`), in kJ/mol.
    const REFERENCES: [(&str, f64); 4] = [
        ("Lys1", -2401.2),
        ("Lys2", -2161.8),
        ("Lys4", -2089.0),
        ("Lys8", -2065.5),
    ];
    let mut ours = Vec::with_capacity(REFERENCES.len());
    for (label, ref_kj) in &REFERENCES {
        let (surface, charges) = load_lysozyme_at(&label[3..]);
        let e_kj = solve_and_report(label, &surface, &charges);
        let rel = (e_kj - ref_kj).abs() / ref_kj.abs();
        eprintln!("  pygbe = {ref_kj:+.2} kJ/mol, rel = {:.2}%", rel * 100.0);
        assert!(
            rel < 0.02,
            "{label}: rel.err vs pygbe {:.2}% > 2 %",
            rel * 100.0
        );
        ours.push(e_kj);
    }
    // why: the published pygbe series is monotone (each refinement
    // steps E_solv *up* toward the continuum limit). Our series
    // should exhibit the same pattern — if some refinement ever
    // reverses direction, something about the BEM or the mesh
    // handling has regressed.
    for pair in ours.windows(2) {
        assert!(
            pair[1] > pair[0] - 1.0,
            "non-monotone convergence: {pair:?} kJ/mol"
        );
    }
}
