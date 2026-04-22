//! Phase-1 milestone: BEM reaction-field potential must match the Kirkwood
//! 1934 analytical result on a sphere for a point charge in the exterior.
//!
//! Setup (from `bem_pb_plan.md §5, step 8`):
//! - Sphere radius `a = 10 Å`, ε_in = 2, ε_out = 80, κ = 0.
//! - Single unit source at `r_1 = (12, 0, 0)` Å.
//! - Evaluation point `r_2 = (6, 6√3, 0)` Å (same distance, 60° apart).
//! - Mesh: hexasphere `subdivisions ∈ {1, 3, 7}` → 80 / 320 / 1280 triangles.
//!
//! Why single-source instead of pair: `φ_rf(r_2; r_1)` from Kirkwood is
//! well-defined per unit source; our BEM output at `r_2` from a one-source
//! solve is the same quantity. Pair energies W_12 then follow as
//! `q_1 · q_2 · φ_rf`. Solving with both charges as sources mixes in the
//! Born self-energy of each charge and complicates the comparison.
//!
//! Acceptance: relative error < 1 % at subdivisions = 7.

#![cfg(feature = "validation")]

use selfie::analytical::kirkwood::reaction_field_potential_unit_source;
use selfie::{BemSolution, Dielectric, Surface};

const A: f64 = 10.0; // sphere radius (Å)
const R: f64 = 13.0; // charge/eval radius (Å) — 3 Å from surface. Close
// enough to give a physically meaningful reaction field
// (~0.3 kcal/mol pair-like scale) but still comfortably
// larger than the panel edge ~ 1 Å at subdiv = 7.
const EPS_IN: f64 = 2.0;
const EPS_OUT: f64 = 80.0;

fn source() -> [f64; 3] {
    [R, 0.0, 0.0]
}

fn eval_point() -> [f64; 3] {
    // γ = 30°: chord |r_i − r_j| = 2 R sin(15°) ≈ 6.7 Å — salt-bridge-scale
    // pair, both charges 3 Å from the dielectric surface.
    [R * (3f64.sqrt() / 2.0), R * 0.5, 0.0]
}

fn bem_phi_rf(subdivisions: usize) -> f64 {
    let surface = Surface::icosphere(A, subdivisions);
    let media = Dielectric::continuum(EPS_IN, EPS_OUT);
    let positions = [source()];
    let values = [1.0_f64];
    let sol = BemSolution::solve(&surface, media, &positions, &values).unwrap();
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
    for s in [1usize, 3, 7, 15] {
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

    // why: collocation-with-Gauss-quadrature is O(h²); since triangle area
    // ~ h² and n_t ~ 1/h², error should scale roughly like 1/n_t. Check the
    // slope of log|err| vs log(1/√n_t) is close to 2 so silent regressions
    // to a worse-order method are caught.
    let (n0, e0) = errors[0];
    let (n2, e2) = errors[2];
    let slope = (e0.ln() - e2.ln()) / ((n2 as f64).ln() - (n0 as f64).ln()) * 2.0;
    eprintln!("Convergence slope (error ∝ h^slope): {slope:.2}");
    assert!(
        slope > 1.2,
        "Convergence slope too shallow: {slope:.2} (expected ≳ 2 for O(h²))"
    );
}
