//! Phase-1 milestone: BEM reaction-field potential must match the Kirkwood
//! 1934 analytical result on a sphere for a point charge in the exterior.
//!
//! Acceptance: relative error < 1 % at subdivisions = 7 (n_t = 1280) for the
//! salt-bridge-scale geometry defined in `tests/common`.

#![cfg(feature = "validation")]

mod common;

use common::{A, EPS_IN, EPS_OUT, eval_point, source};
use selfie::analytical::kirkwood::reaction_field_potential_unit_source;
use selfie::{BemSolution, ChargeSide, Dielectric, Surface};

fn bem_phi_rf(subdivisions: usize) -> f64 {
    let surface = Surface::icosphere(A, subdivisions);
    let media = Dielectric::continuum(EPS_IN, EPS_OUT);
    let sol =
        BemSolution::solve(&surface, media, ChargeSide::Exterior, &[source()], &[1.0]).unwrap();
    sol.reaction_field_at(eval_point())
}

fn kirkwood_phi_rf() -> f64 {
    reaction_field_potential_unit_source(source(), eval_point(), A, EPS_IN, EPS_OUT, 80)
}

#[test]
fn convergence_to_kirkwood() {
    let reference = kirkwood_phi_rf();
    eprintln!("Kirkwood φ_rf(r_2; r_1) = {reference:.9e}  (reduced units, e/Å)");

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
        "Phase-1 acceptance failed: rel. err at n_t=1280 is {:.3e} (need < 1%)",
        rel_1280
    );

    // why: collocation with Gauss quadrature is O(h²); since n_t ~ 1/h² the
    // error scales like 1/n_t. The slope check guards against silent
    // regressions to a worse-order method.
    let (n0, e0) = errors[0];
    let (n2, e2) = errors[2];
    let slope = (e0.ln() - e2.ln()) / ((n2 as f64).ln() - (n0 as f64).ln()) * 2.0;
    eprintln!("Convergence slope (error ∝ h^slope): {slope:.2}");
    assert!(
        slope > 1.2,
        "Convergence slope too shallow: {slope:.2} (expected ≳ 2 for O(h²))"
    );
}
