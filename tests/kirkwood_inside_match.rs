//! Inside-charge milestone: BEM reaction-field must match Kirkwood's
//! classical 1934 interior-charge analytical solution. This is the
//! production-relevant geometry for MC pKa titration of globular proteins,
//! where ionizable groups sit inside the low-ε protein interior.
//!
//! Geometry: sphere a = 10 Å; two unit charges at r = 7 Å (3 Å below
//! the surface), γ = 30° apart (chord ≈ 3.6 Å — a realistic salt-bridge
//! inside a small protein). ε_in = 2, ε_out = 80, κ = 0 (interior is
//! always Laplace — κ applies only to the exterior solvent).
//!
//! Acceptance: relative error < 1 % at subdivisions = 7 (n_t = 1280).

#![cfg(feature = "validation")]

use selfie::analytical::kirkwood_inside::reaction_field_potential_unit_source as phi_rf_analytical;
use selfie::{BemSolution, ChargeSide, Dielectric, Surface};

const A: f64 = 10.0;
const R: f64 = 7.0; // 3 Å beneath the surface
const EPS_IN: f64 = 2.0;
const EPS_OUT: f64 = 80.0;

fn source() -> [f64; 3] {
    [R, 0.0, 0.0]
}

fn eval_point() -> [f64; 3] {
    // γ = 30° → pair chord 2R sin(15°) ≈ 3.62 Å.
    [R * (3f64.sqrt() / 2.0), R * 0.5, 0.0]
}

fn bem_phi_rf(subdivisions: usize) -> f64 {
    let surface = Surface::icosphere(A, subdivisions);
    let media = Dielectric::continuum(EPS_IN, EPS_OUT);
    let sol =
        BemSolution::solve(&surface, media, ChargeSide::Interior, &[source()], &[1.0]).unwrap();
    sol.reaction_field_at(eval_point())
}

#[test]
fn convergence_to_kirkwood_inside() {
    let reference = phi_rf_analytical(source(), eval_point(), A, EPS_IN, EPS_OUT, 80);
    eprintln!("Kirkwood inside φ_rf(r_2; r_1) = {reference:.9e}  (reduced units, e/Å)");

    let mut errors = Vec::new();
    for s in [1usize, 3, 7] {
        let bem = bem_phi_rf(s);
        let n_t = 20 * (s + 1).pow(2);
        let rel = (bem - reference).abs() / reference.abs();
        eprintln!("  subdiv = {s:2}  n_t = {n_t:4}  BEM = {bem:.9e}  rel.err = {rel:.3e}");
        errors.push((n_t, rel));
    }

    let (n_t_1280, rel_1280) = errors[2];
    assert_eq!(n_t_1280, 1280);
    assert!(
        rel_1280 < 0.01,
        "Inside-charge acceptance failed: rel. err at n_t = 1280 is {:.3e} (need < 1 %)",
        rel_1280
    );
}

#[test]
fn like_charges_inside_are_stabilized_by_solvent() {
    // Physics check: for ε_in < ε_out, the reaction field at r_j from a
    // unit source at r_i inside the cavity has the *opposite sign* to
    // φ_source. i.e. like-charge pair repulsion is screened. Also
    // confirms sign convention of the interior evaluator.
    let phi = bem_phi_rf(7);
    assert!(
        phi < 0.0,
        "expected φ_rf < 0 for interior pair with ε_in < ε_out, got {phi}"
    );
}
