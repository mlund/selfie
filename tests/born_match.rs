//! Born (κ = 0) and Debye-Hückel (κ > 0) closed-form checks.
//!
//! Place a unit charge at the centre of a dielectric sphere and ask the
//! BEM for the reaction-field potential at the origin. By spherical
//! symmetry the analytical answer collapses to a single-line formula (see
//! `selfie::analytical::born`) with no Bessel or Legendre machinery —
//! this is the tightest possible sanity check on the whole interior-
//! charge pipeline (assembly RHS, Juffer block, interior evaluator).
//!
//! Acceptance: relative error < 0.5 % at subdivisions = 7 (n_t = 1280).

#![cfg(feature = "validation")]

use selfie::analytical::born::reaction_field_at_center;
use selfie::{BemSolution, ChargeSide, Dielectric, Surface};

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

    let mut finest_rel = f64::INFINITY;
    for s in [1usize, 3, 7] {
        let bem = bem_phi_rf_center(s, kappa);
        let n_t = 20 * (s + 1).pow(2);
        let rel = (bem - reference).abs() / reference.abs();
        eprintln!("  subdiv = {s:2}  n_t = {n_t:4}  BEM = {bem:.9e}  rel.err = {rel:.3e}");
        finest_rel = rel;
    }
    assert!(
        finest_rel < 5e-3,
        "{label}: rel. err at finest = {finest_rel:.3e}"
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
