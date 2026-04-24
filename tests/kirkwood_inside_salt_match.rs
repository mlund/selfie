//! Inside-charge + salt validation: BEM reaction-field energy for point
//! charges inside a sphere with salty solvent must match the analytical
//! multipole series (axisymmetric, charges on z-axis; formula ported
//! from pygbe's `an_P`). Closes the 2 × 2 `{inside, outside} ×
//! {κ=0, κ>0}` coverage grid.
//!
//! Acceptance: rel. error < 2 % at subdivisions = 7 (n_t = 1280), with a
//! rate check that the error reduces ≳ 3× per 4× mesh refinement
//! (O(h²) collocation). The absolute tolerance is looser than our other
//! sphere tests because two off-centre charges with salt carry more
//! multipole content per unit volume than the centred / single-charge
//! geometries elsewhere.

#![cfg(feature = "validation")]

use selfie::analytical::kirkwood_inside_salt::solvation_energy_z_axis;
use selfie::{BemSolution, ChargeSide, Dielectric, Surface};

const A: f64 = 10.0;
const EPS_IN: f64 = 2.0;
const EPS_OUT: f64 = 80.0;
const KAPPA: f64 = 1.0 / 9.0;

struct Params {
    r_sphere: f64,
    eps_in: f64,
    eps_out: f64,
    kappa: f64,
}

fn bem_solvation_with(p: &Params, subdivisions: usize, charges_on_z: &[(f64, f64)]) -> f64 {
    let surface = Surface::icosphere(p.r_sphere, subdivisions);
    let media = Dielectric::continuum_with_salt(p.eps_in, p.eps_out, p.kappa);
    let positions: Vec<[f64; 3]> = charges_on_z.iter().map(|&(_, z)| [0.0, 0.0, z]).collect();
    let values: Vec<f64> = charges_on_z.iter().map(|&(q, _)| q).collect();
    let sol =
        BemSolution::solve(&surface, media, ChargeSide::Interior, &positions, &values).unwrap();
    let mut phi = vec![0.0_f64; positions.len()];
    sol.reaction_field_at_many(&positions, &mut phi).unwrap();
    0.5 * positions
        .iter()
        .zip(&phi)
        .enumerate()
        .map(|(i, (_, &p))| values[i] * p)
        .sum::<f64>()
}

#[test]
fn convergence_to_inside_salt_multipole() {
    // Two unlike charges on z-axis, both 3 Å beneath the surface of a
    // 10 Å sphere, 4 Å apart along z — a buried salt-bridge with
    // physiological-scale salt outside.
    let charges = [(1.0, 5.0), (-1.0, -5.0)];
    let reference = solvation_energy_z_axis(&charges, EPS_IN, EPS_OUT, A, A, KAPPA, 60);
    eprintln!("Kirkwood-inside+salt U_solv = {reference:.9e}  (reduced e²/Å)");

    let p = Params {
        r_sphere: A,
        eps_in: EPS_IN,
        eps_out: EPS_OUT,
        kappa: KAPPA,
    };
    let mut errors = Vec::new();
    for s in [1usize, 3, 7] {
        let bem = bem_solvation_with(&p, s, &charges);
        let n_t = 20 * (s + 1).pow(2);
        let rel = (bem - reference).abs() / reference.abs();
        eprintln!("  subdiv = {s:2}  n_t = {n_t:4}  BEM = {bem:.9e}  rel.err = {rel:.3e}");
        errors.push(rel);
    }

    let finest_rel = errors[2];
    assert!(finest_rel < 0.02, "rel.err at n_t = 1280: {finest_rel:.3e}");

    // why: rate check guards against O(h²) → O(h) regressions. Each 4×
    // mesh refinement should reduce the error by at least 3× for a
    // correctly-implemented O(h²) method (the 4× ideal leaves margin).
    let rate_coarse = errors[0] / errors[1];
    let rate_fine = errors[1] / errors[2];
    assert!(
        rate_coarse > 3.0 && rate_fine > 3.0,
        "convergence rate too slow: {rate_coarse:.2}, {rate_fine:.2}"
    );
}

/// Reproduces pygbe's `examples/sphere` validation geometry: R = 4 Å,
/// ε_in = 4, ε_out = 80, κ = 0.125 Å⁻¹ (Debye length ≈ 8 Å), a single
/// unit charge 2 Å from centre (2 Å beneath the surface). pygbe
/// validates the BEM output against the same Kirkwood + Debye-Hückel
/// series we've ported. By spherical symmetry the self-solvation of an
/// off-axis charge depends only on |r|, so the on-axis test suffices.
///
/// Reference: <https://github.com/pygbe/pygbe> — examples/sphere.
#[test]
fn reproduces_pygbe_sphere_example_geometry() {
    let p = Params {
        r_sphere: 4.0,
        eps_in: 4.0,
        eps_out: 80.0,
        kappa: 0.125,
    };
    let charges = [(1.0_f64, 2.0_f64)];
    let reference = solvation_energy_z_axis(
        &charges, p.eps_in, p.eps_out, p.r_sphere, p.r_sphere, p.kappa, 60,
    );
    eprintln!(
        "pygbe sphere geometry: U_self = {reference:.6e} e²/Å \
         = {:.3} kJ/mol",
        selfie::units::to_kJ_per_mol(reference)
    );

    let bem = bem_solvation_with(&p, 7, &charges);
    let rel = (bem - reference).abs() / reference.abs();
    eprintln!("  BEM subdiv = 7 (n_t = 1280): U = {bem:.6e}, rel.err = {rel:.3e}");
    assert!(rel < 0.02, "pygbe-sphere: rel.err = {rel:.3e}");
}
