//! Real-protein I/O smoke test on pygbe's lysozyme single-surface setup.
//!
//! pygbe's `tests/convergence_tests/input_files/lys_single_1.config`
//! solves lysozyme with a single closed protein surface (no Stern layer,
//! no internal cavities) against bulk water at κ = 0.125 Å⁻¹. That's
//! exactly our physics setup — but the Lys1 mesh has 14,398 faces,
//! which blows up to a ~6.6 GB dense LU matrix. Until we have a
//! matrix-free iterative solver, we can't realistically run the full
//! solve in a default test.
//!
//! What this test *does* verify, end-to-end on a real protein:
//!
//! - MSMS `.vert`/`.face` reader handles the large file and the
//!   triangle-soup dedup collapses correctly.
//! - `Surface::from_mesh` manifoldness check passes on the SES mesh
//!   (every directed edge paired with its opposite).
//! - PQR reader parses the ~1,300-atom charge file.
//! - `Surface::classify_charges` correctly places all atoms interior.
//!
//! That's the file-I/O + geometry-sanity pipeline on a realistic
//! workload. The actual BEM solve on this mesh is gated behind a
//! separate opt-in test (not in this file) once iterative solvers land.

#![cfg(feature = "validation")]

use selfie::io::{read_msms, read_pqr};
use selfie::{ChargeSide, Surface};
use std::path::PathBuf;

fn fixture(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pygbe_lys")
        .join(sub)
}

#[test]
fn lysozyme_single_surface_pipeline() {
    let vert = fixture("Lys1.vert");
    let face = fixture("Lys1.face");
    let pqr = fixture("built_parse.pqr");

    let (vertices, faces) = read_msms(&vert, &face).expect("read_msms");
    // MSMS dedup: Lys1 ships shared-vertex (not triangle-soup), so the
    // raw counts already reflect the topological structure.
    assert_eq!(faces.len(), 14_398);
    assert_eq!(vertices.len(), 7_201);

    // Surface::from_mesh runs the closed-orientable-2-manifold check
    // and the per-face degeneracy checks. If this passes, the mesh is
    // topologically sound for BEM use.
    let surface = Surface::from_mesh(&vertices, &faces)
        .expect("Lys1 mesh should be a closed orientable 2-manifold");
    assert_eq!(surface.num_faces(), 14_398);

    let charges = read_pqr(&pqr).expect("read_pqr");
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
        faces.len(),
        charges.values.len(),
        total
    );
}
