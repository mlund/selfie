//! `LinearResponse` against the same closed-form oracles that
//! already validate `BemSolution` — Born, interior Kirkwood,
//! interior Kirkwood + salt. Each test mirrors the tolerance the
//! underlying `BemSolution` match achieves at the same mesh, because
//! the basis path inherits its discretisation error from the same
//! per-site solves.

#![cfg(feature = "validation")]

use bemtzmann::analytical::born::born_self_energy;
use bemtzmann::analytical::kirkwood_inside::reaction_field_potential_unit_source as phi_rf_analytical;
use bemtzmann::analytical::kirkwood_inside_salt::solvation_energy_z_axis;
use bemtzmann::{ChargeSide, Dielectric, LinearResponse, Surface};

const A: f64 = 10.0;
const EPS_IN: f64 = 2.0;
const EPS_OUT: f64 = 80.0;

#[test]
fn matches_born_at_kappa_zero() {
    // One unit charge at the centre of a radius-10 sphere; basis
    // solvation energy must reproduce the Born closed form
    // `½ q²/a · (1/ε_out − 1/ε_in)` to the same tolerance the
    // existing `tests/born_match.rs` achieves with BemSolution.
    let surface = Surface::icosphere(A, 7);
    let media = Dielectric::continuum(EPS_IN, EPS_OUT);
    let basis =
        LinearResponse::precompute(&surface, media, ChargeSide::Interior, &[[0.0, 0.0, 0.0]])
            .unwrap();
    let bem = basis.solvation_energy(&[1.0]).unwrap();
    let reference = born_self_energy(1.0, A, EPS_IN, EPS_OUT, 0.0);
    let rel = (bem - reference).abs() / reference.abs();
    eprintln!("Born: BEM = {bem:.9e}, reference = {reference:.9e}, rel = {rel:.3e}");
    assert!(rel < 5e-3, "Born match: rel = {rel:.3e}");
}

#[test]
fn matches_debye_huckel_at_physiological_salt() {
    // κ > 0 sibling of the Born test. Centred charge collapses the
    // Kirkwood series to the Debye-Hückel formula in `born::`.
    let kappa = 1.0 / 14.0;
    let surface = Surface::icosphere(A, 7);
    let media = Dielectric::continuum_with_salt(EPS_IN, EPS_OUT, kappa);
    let basis =
        LinearResponse::precompute(&surface, media, ChargeSide::Interior, &[[0.0, 0.0, 0.0]])
            .unwrap();
    let bem = basis.solvation_energy(&[1.0]).unwrap();
    let reference = born_self_energy(1.0, A, EPS_IN, EPS_OUT, kappa);
    let rel = (bem - reference).abs() / reference.abs();
    eprintln!("Debye-Hückel: BEM = {bem:.9e}, reference = {reference:.9e}, rel = {rel:.3e}");
    assert!(rel < 5e-3, "DH match: rel = {rel:.3e}");
}

#[test]
fn matches_kirkwood_inside_multipair() {
    // Three interior charges inside the radius-10 sphere; build the
    // full bilinear energy from the Kirkwood interior Legendre series
    // (same formula the existing `BemSolution`-against-Kirkwood test
    // uses) and require the basis to reproduce it.
    let sites = [
        [7.0, 0.0, 0.0],
        [7.0 * (3f64.sqrt() / 2.0), 7.0 * 0.5, 0.0],
        [0.0, 0.0, 6.5],
    ];
    let charges = [1.0, -0.8, 0.5];

    let surface = Surface::icosphere(A, 7);
    let media = Dielectric::continuum(EPS_IN, EPS_OUT);
    let basis = LinearResponse::precompute(&surface, media, ChargeSide::Interior, &sites).unwrap();
    let bem = basis.solvation_energy(&charges).unwrap();

    let mut reference = 0.0;
    for i in 0..sites.len() {
        for j in 0..sites.len() {
            reference += charges[i]
                * charges[j]
                * phi_rf_analytical(sites[i], sites[j], A, EPS_IN, EPS_OUT, 80);
        }
    }
    reference *= 0.5;

    let rel = (bem - reference).abs() / reference.abs();
    eprintln!("Kirkwood inside: BEM = {bem:.9e}, ref = {reference:.9e}, rel = {rel:.3e}");
    // why: the diagonal i == i terms evaluate the reaction field at
    // a source's own position, whose BEM discretisation error is
    // larger than at off-source points (the same 2 %-class
    // tolerance the `kirkwood_inside_salt_match.rs` self-evaluation
    // test lives at).
    assert!(rel < 0.02, "Kirkwood inside: rel = {rel:.3e}");
}

#[test]
fn matches_kirkwood_inside_salt_z_axis() {
    // Salt-bridge geometry on the z-axis — the oracle's axisymmetry
    // constraint. Mirrors `tests/kirkwood_inside_salt_match.rs`'s
    // setup and tolerance (2 % at subdiv 7).
    let kappa = 1.0 / 9.0;
    let charges_on_z = [(1.0, 5.0), (-1.0, -5.0)];
    let sites: Vec<[f64; 3]> = charges_on_z.iter().map(|&(_, z)| [0.0, 0.0, z]).collect();
    let values: Vec<f64> = charges_on_z.iter().map(|&(q, _)| q).collect();

    let surface = Surface::icosphere(A, 7);
    let media = Dielectric::continuum_with_salt(EPS_IN, EPS_OUT, kappa);
    let basis = LinearResponse::precompute(&surface, media, ChargeSide::Interior, &sites).unwrap();
    let bem = basis.solvation_energy(&values).unwrap();
    let reference = solvation_energy_z_axis(&charges_on_z, EPS_IN, EPS_OUT, A, A, kappa, 60);

    let rel = (bem - reference).abs() / reference.abs();
    eprintln!("Kirkwood inside+salt: BEM = {bem:.9e}, ref = {reference:.9e}, rel = {rel:.3e}");
    assert!(rel < 0.02, "Kirkwood inside+salt: rel = {rel:.3e}");
}
