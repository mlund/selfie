//! Cross-BEM validation against pygbe's stored result for its
//! `examples/sphere` regression case.
//!
//! pygbe (<https://github.com/pygbe/pygbe>) is an independent
//! boundary-element PB solver (Python + CUDA). Its regression
//! `tests/sphere.pickle` records the solvation energy for a single
//! off-centre unit charge in a 4 Å cavity with κ = 0.125 Å⁻¹ salt at
//! 512 mesh elements:
//!
//!     E_solv = −13.458119761457832 kcal/mol
//!
//! By spherical symmetry the self-solvation of an off-axis source at
//! |r| = 2 equals the on-axis source at z = 2, so our axisymmetric
//! analytical applies. At matching mesh resolution (hexasphere
//! subdiv = 4 → 500 triangles, vs pygbe's 512) the two BEMs should
//! agree well inside the discretisation tolerance of the analytical.

#![cfg(feature = "validation")]

use selfie::analytical::kirkwood_inside_salt::solvation_energy_z_axis;
use selfie::units::to_kcal_per_mol;
use selfie::{BemSolution, ChargeSide, Dielectric, Surface};

const PYGBE_E_SOLV_KCAL: f64 = -13.458119761457832;

#[test]
fn bem_matches_pygbe_sphere_result_at_equivalent_mesh() {
    let r_sphere = 4.0;
    let eps_in = 4.0;
    let eps_out = 80.0;
    let kappa = 0.125;
    let charges_z = [(1.0_f64, 2.0_f64)];
    let positions = [[0.0, 0.0, 2.0]];
    let values = [1.0_f64];

    let analytical =
        solvation_energy_z_axis(&charges_z, eps_in, eps_out, r_sphere, r_sphere, kappa, 60);

    // hexasphere subdiv = 4 → 500 triangles, closest match to pygbe's 512.
    let surface = Surface::icosphere(r_sphere, 4);
    let media = Dielectric::continuum_with_salt(eps_in, eps_out, kappa);
    let sol =
        BemSolution::solve(&surface, media, ChargeSide::Interior, &positions, &values).unwrap();
    let u = 0.5 * values[0] * sol.reaction_field_at(positions[0]);
    let u_kcal = to_kcal_per_mol(u);

    let rel_vs_pygbe = (u_kcal - PYGBE_E_SOLV_KCAL).abs() / PYGBE_E_SOLV_KCAL.abs();
    let rel_vs_analytical = (u - analytical).abs() / analytical.abs();
    eprintln!(
        "Analytical:   {:+.4} kcal/mol\n\
         pygbe (512):  {:+.4} kcal/mol\n\
         selfie (500): {:+.4} kcal/mol  \
         (vs pygbe: {:.2e}, vs analytical: {:.2e})",
        to_kcal_per_mol(analytical),
        PYGBE_E_SOLV_KCAL,
        u_kcal,
        rel_vs_pygbe,
        rel_vs_analytical
    );

    // why: 0.5 % threshold. The two BEMs use different meshes (500 vs
    // 512 triangles), different quadratures and different solvers, so
    // some discretisation spread is expected. If we ever drift further
    // than this at the same mesh resolution, the physics is diverging.
    assert!(
        rel_vs_pygbe < 5e-3,
        "selfie vs pygbe: rel.err = {rel_vs_pygbe:.3e}"
    );
}
