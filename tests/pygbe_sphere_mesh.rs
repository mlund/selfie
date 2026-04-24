//! Real-mesh round-trip against pygbe's `examples/sphere`.
//!
//! Loads pygbe's `.vert` + `.face` mesh and `.pqr` charge file (copied
//! into `tests/data/pygbe_sphere/`), classifies the charge side via
//! `Surface::classify_charges`, runs our BEM solver, and compares the
//! solvation energy to pygbe's stored regression:
//!
//!     pygbe/tests/sphere.pickle  →  E_solv = −13.458119761457832 kcal/mol
//!                                           = −56.294777697... kJ/mol

#![cfg(feature = "validation")]

use selfie::io::{read_msms, read_pqr};
use selfie::units::to_kJ_per_mol;
use selfie::{BemSolution, Dielectric, Surface};
use std::path::PathBuf;

const PYGBE_E_SOLV_KJ: f64 = -56.294777697138567;

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
    let u_kj = to_kJ_per_mol(u);

    let rel = (u_kj - PYGBE_E_SOLV_KJ).abs() / PYGBE_E_SOLV_KJ.abs();
    eprintln!(
        "file I/O + BEM: U_solv = {u_kj:+.4} kJ/mol  \
         (pygbe: {PYGBE_E_SOLV_KJ:+.4}, rel = {rel:.3e})"
    );
    assert!(rel < 5e-3, "selfie vs pygbe: rel.err = {rel:.3e}");
}

/// Exterior-charge coverage on the pygbe sphere mesh. Closes the
/// `{mesh = pygbe-mesh} × {charge-side = Exterior}` hole in the
/// existing test matrix — every other exterior-charge regression
/// uses `Surface::icosphere`, not an externally-built MSMS mesh, so
/// an I/O-path regression that only bit the Yukawa RHS would
/// otherwise slip through.
///
/// Geometry: pygbe's R = 4 Å sphere; one unit charge at `[6, 0, 0]`
/// (2 Å outside the surface). Evaluate φ_rf at `[0, 6, 0]` (also
/// outside, γ = 90° from the source). Compare against the classical
/// Kirkwood-exterior analytical (κ = 0) — same oracle as
/// `tests/kirkwood_match.rs`, now exercised on a real MSMS mesh.
#[test]
fn exterior_charge_on_pygbe_sphere_matches_kirkwood() {
    use selfie::ChargeSide;
    use selfie::analytical::kirkwood::reaction_field_potential_unit_source as phi_rf_analytical;

    let vert = test_data("sphere500_R4.vert");
    let face = test_data("sphere500_R4.face");
    let (vertices, faces) = read_msms(&vert, &face).expect("failed to read MSMS mesh");
    let surface = Surface::from_mesh(&vertices, &faces).expect("from_mesh rejected the mesh");

    let a = 4.0;
    let eps_in = 4.0;
    let eps_out = 80.0;
    let source = [6.0, 0.0, 0.0];
    let eval = [0.0, 6.0, 0.0];

    let media = Dielectric::continuum(eps_in, eps_out);
    let sol = BemSolution::solve(&surface, media, ChargeSide::Exterior, &[source], &[1.0])
        .expect("exterior BemSolution::solve failed");
    let bem = sol.reaction_field_at(eval);
    let reference = phi_rf_analytical(source, eval, a, eps_in, eps_out, 80);

    let rel = (bem - reference).abs() / reference.abs();
    eprintln!("exterior on pygbe sphere: BEM = {bem:.9e}, Kirkwood = {reference:.9e}, rel = {rel:.3e}");
    // why: the pygbe sphere at 500 triangles is coarser than our
    // subdiv = 7 icosphere baseline (1280). The existing exterior-
    // Kirkwood gate at subdiv = 3 (n_t = 320) holds to < 5 %; keep
    // the same envelope here.
    assert!(rel < 5e-2, "exterior on pygbe sphere: rel = {rel:.3e}");
}
