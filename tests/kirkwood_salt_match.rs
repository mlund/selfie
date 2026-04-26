//! κ > 0 validation: BEM reaction-field must match the salted Kirkwood
//! analytical result on a sphere. Physiological target: κ = 1/9 Å⁻¹
//! (Debye length 9 Å ≈ 115 mM 1:1 salt — near physiological).
//!
//! Acceptance at subdivisions = 7 (n_t = 1280): relative error < 1 %.

#![cfg(feature = "validation")]

mod common;

use common::{A, EPS_IN, EPS_OUT, eval_point, source};
use bemtzmann::analytical::kirkwood_salt::reaction_field_potential_unit_source as phi_rf_analytical;
use bemtzmann::{BemSolution, ChargeSide, Dielectric, Surface};

const KAPPA: f64 = 1.0 / 9.0;

fn bem_phi_rf(subdivisions: usize, kappa: f64) -> f64 {
    let surface = Surface::icosphere(A, subdivisions);
    let media = Dielectric::continuum_with_salt(EPS_IN, EPS_OUT, kappa);
    let sol =
        BemSolution::solve(&surface, media, ChargeSide::Exterior, &[source()], &[1.0]).unwrap();
    sol.reaction_field_at(eval_point())
}

#[test]
fn convergence_to_kirkwood_salt_physiological() {
    let reference = phi_rf_analytical(source(), eval_point(), A, EPS_IN, EPS_OUT, KAPPA, 80);
    eprintln!("Kirkwood+salt φ_rf (κ = 1/9 Å⁻¹) = {reference:.9e}");

    let mut errors = Vec::new();
    for s in [1usize, 3, 7] {
        let bem = bem_phi_rf(s, KAPPA);
        let n_t = 20 * (s + 1).pow(2);
        let rel = (bem - reference).abs() / reference.abs();
        eprintln!("  subdiv = {s:2}  n_t = {n_t:4}  BEM = {bem:.9e}  rel.err = {rel:.3e}");
        errors.push((n_t, rel));
    }

    let (n_t_1280, rel_1280) = errors[2];
    assert_eq!(n_t_1280, 1280);
    assert!(
        rel_1280 < 0.01,
        "κ > 0 acceptance failed: rel. err at n_t = 1280 is {:.3e} (need < 1 %)",
        rel_1280
    );
}

#[test]
fn screening_reduces_magnitude_vs_salt_free() {
    // For a close pair (≈ 6.7 Å apart, 3 Å above the dielectric surface) salt
    // shields both the direct and reaction-field responses, so
    // |φ_rf(κ > 0)| < |φ_rf(κ = 0)|. At wide separations the inequality is
    // non-monotonic, so this test is specific to the salt-bridge regime.
    let salted = bem_phi_rf(7, KAPPA);
    let salt_free = bem_phi_rf(7, 0.0);
    assert!(
        salted.abs() < salt_free.abs(),
        "salt should shield: |salted|={:e} !< |salt-free|={:e}",
        salted.abs(),
        salt_free.abs()
    );
}
