//! Convergence study for the Gaussian-surface mesher: a single atom at the
//! origin is meshed at three grid spacings, and the BEM reaction-field
//! potential at the centre is compared to the analytical Born formula.
//!
//! With decay = 1 and isolevel = 1, the single-atom Gaussian surface is the
//! sphere of radius `R` exactly — so the same Born reference used for the
//! icosphere applies. Marching cubes is first-order in the grid spacing on
//! curved surfaces, so the test asserts convergence rather than a specific
//! rate.

#![cfg(all(feature = "mesh", feature = "validation"))]

use bemtzmann::analytical::born::reaction_field_at_center;
use bemtzmann::{BemSolution, ChargeSide, Dielectric, Surface};

const A: f64 = 10.0;
const EPS_IN: f64 = 2.0;
const EPS_OUT: f64 = 80.0;
const CENTER: [f64; 3] = [0.0; 3];

fn bem_phi_rf_at_centre(spacing: f64) -> f64 {
    let surface = Surface::from_atoms_gaussian(&[CENTER], &[A], spacing)
        .expect("Gaussian mesh on isolated atom");
    let media = Dielectric::continuum(EPS_IN, EPS_OUT);
    let sol = BemSolution::solve(&surface, media, ChargeSide::Interior, &[CENTER], &[1.0]).unwrap();
    sol.reaction_field_at(CENTER)
}

#[test]
fn gaussian_mesh_converges_to_born() {
    let reference = reaction_field_at_center(A, EPS_IN, EPS_OUT, 0.0);
    eprintln!("Born φ_rf(0) = {reference:.9e}  (reduced units, e/Å)");

    let mut errors = Vec::new();
    for spacing in [1.0_f64, 0.5, 0.25] {
        let bem = bem_phi_rf_at_centre(spacing);
        let rel = (bem - reference).abs() / reference.abs();
        eprintln!("  spacing = {spacing:5.3} Å   BEM = {bem:.9e}   rel.err = {rel:.3e}");
        errors.push(rel);
    }

    assert!(
        errors[2] < 0.05,
        "rel. error at finest spacing = {:.3e} (need < 5%)",
        errors[2]
    );
    // why: marching cubes is first-order in h on a curved surface, so a
    // 4× refinement (1.0 → 0.25 Å) should drop the error by ~4×. We allow a
    // factor of 2 to absorb sub-grid alignment effects and BEM quadrature
    // noise — the assertion catches order-of-convergence regressions while
    // tolerating mild non-monotonicity.
    let ratio = errors[0] / errors[2];
    assert!(
        ratio > 2.0,
        "spacing 1.0 → 0.25 should drop error by > 2× (got {ratio:.2}×)"
    );
}
