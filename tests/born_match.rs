//! Born (κ = 0) and Debye-Hückel (κ > 0) closed-form checks.
//!
//! Place a unit charge at the centre of a dielectric sphere and ask the
//! BEM for the reaction-field potential at the origin. By spherical
//! symmetry the analytical answer collapses to a single-line formula (see
//! `bemtzmann::analytical::born`) with no Bessel or Legendre machinery —
//! this is the tightest possible sanity check on the whole interior-
//! charge pipeline (assembly RHS, Juffer block, interior evaluator).
//!
//! Acceptance: relative error < 0.5 % at subdivisions = 7 (n_t = 1280).

#![cfg(feature = "validation")]

use bemtzmann::analytical::born::reaction_field_at_center;
use bemtzmann::{BemSolution, ChargeSide, Dielectric, Surface};

const A: f64 = 10.0;
const EPS_IN: f64 = 2.0;
const EPS_OUT: f64 = 80.0;
const CENTER: [f64; 3] = [0.0; 3];

fn bem_phi_rf_center(subdivisions: usize, kappa: f64) -> f64 {
    let surface = Surface::icosphere(A, subdivisions);
    let media = Dielectric::continuum_with_salt(EPS_IN, EPS_OUT, kappa);
    let sol = BemSolution::solve(&surface, media, ChargeSide::Interior, &[CENTER], &[1.0]).unwrap();
    sol.reaction_field_at(CENTER)
}

fn check_convergence(label: &str, kappa: f64) {
    let reference = reaction_field_at_center(A, EPS_IN, EPS_OUT, kappa);
    eprintln!("{label} φ_rf(0) = {reference:.9e}  (reduced units, e/Å)");

    let mut errors = Vec::new();
    for s in [1usize, 3, 7] {
        let bem = bem_phi_rf_center(s, kappa);
        let n_t = 20 * (s + 1).pow(2);
        let rel = (bem - reference).abs() / reference.abs();
        eprintln!("  subdiv = {s:2}  n_t = {n_t:4}  BEM = {bem:.9e}  rel.err = {rel:.3e}");
        errors.push(rel);
    }
    assert!(
        errors[2] < 5e-3,
        "{label}: rel. err at finest = {:.3e}",
        errors[2]
    );

    // why: centroid collocation is O(h²) in panel size. Each 4×
    // panel-count refinement (h halves) should drop the error by
    // at least 3× (4 ideal, margin for roundoff / coarse-grid
    // transients). Catches order-of-convergence regressions.
    let rate_coarse = errors[0] / errors[1];
    let rate_fine = errors[1] / errors[2];
    assert!(
        rate_coarse > 3.0 && rate_fine > 3.0,
        "{label}: convergence too slow (coarse ratio {rate_coarse:.2}, fine ratio {rate_fine:.2})"
    );
}

#[test]
fn born_kappa_zero_matches_closed_form() {
    check_convergence("Born", 0.0);
}

#[test]
fn debye_huckel_matches_closed_form_at_physiological_salt() {
    check_convergence("Debye-Hückel", 1.0 / 14.0);
}
