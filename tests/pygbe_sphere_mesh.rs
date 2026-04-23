//! Real-mesh round-trip against pygbe's `examples/sphere`.
//!
//! Loads pygbe's `.vert` + `.face` mesh and `.pqr` charge file (copied
//! into `tests/data/pygbe_sphere/`), classifies the charge side via
//! `Surface::classify_charges`, runs our BEM solver, and compares the
//! solvation energy to pygbe's stored regression:
//!
//!     pygbe/tests/sphere.pickle  →  E_solv = −13.458119761457832 kcal/mol

#![cfg(feature = "validation")]

use selfie::io::{read_msms, read_pqr};
use selfie::units::to_kcal_per_mol;
use selfie::{BemSolution, Dielectric, Surface};
use std::path::PathBuf;

const PYGBE_E_SOLV_KCAL: f64 = -13.458119761457832;

fn test_data(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pygbe_sphere")
        .join(sub)
}

#[test]
fn bem_reproduces_pygbe_sphere_via_file_io() {
    let vert = test_data("sphere500_R4.vert");
    let face = test_data("sphere500_R4.face");
    let pqr = test_data("offcenter_R2.pqr");

    let (vertices, faces) = read_msms(&vert, &face).expect("failed to read MSMS mesh");
    let surface = Surface::from_mesh(&vertices, &faces).expect("from_mesh rejected the mesh");
    let charges = read_pqr(&pqr).expect("failed to read PQR");

    let side = surface
        .classify_charges(&charges.positions)
        .expect("classify_charges could not determine a single side");

    // pygbe's sphere config: FIELD 2 (exterior) has D = 80, κ = 0.125.
    //                       FIELD 1 (interior) has D_i = 4, κ = 0.
    let media = Dielectric::continuum_with_salt(4.0, 80.0, 0.125);
    let sol = BemSolution::solve(&surface, media, side, &charges.positions, &charges.values)
        .expect("BemSolution::solve failed");

    // Match pygbe's E_solv: (1/2) Σ q_i · φ_rf(r_i).
    let mut phi = vec![0.0_f64; charges.positions.len()];
    sol.reaction_field_at_many(&charges.positions, &mut phi)
        .unwrap();
    let u: f64 = 0.5
        * charges
            .values
            .iter()
            .zip(&phi)
            .map(|(&q, &p)| q * p)
            .sum::<f64>();
    let u_kcal = to_kcal_per_mol(u);

    let rel = (u_kcal - PYGBE_E_SOLV_KCAL).abs() / PYGBE_E_SOLV_KCAL.abs();
    eprintln!(
        "file I/O + BEM: U_solv = {u_kcal:+.4} kcal/mol  \
         (pygbe: {PYGBE_E_SOLV_KCAL:+.4}, rel = {rel:.3e})"
    );
    assert!(rel < 5e-3, "selfie vs pygbe: rel.err = {rel:.3e}");
}
