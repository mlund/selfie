//! `LinearResponse` self-consistency gate on the lysozyme mesh.
//!
//! Builds a small basis on the Lys1 mesh (5 atom positions as sites)
//! and checks that `½ qᵀ G q` agrees with a direct
//! `BemSolution::solve` + `interaction_energy` sum for the same
//! charge configuration. Prints the precompute wallclock so a
//! regression on basis build cost is visible.
//!
//! `#[ignore]`-gated to match `tests/pygbe_lys_mesh.rs`'s
//! `lysozyme_full_solve`.

#![cfg(feature = "validation")]

use selfie::io::{read_msms, read_pqr};
use selfie::{BemSolution, ChargeSide, Dielectric, LinearResponse, Surface};
use std::path::PathBuf;
use std::time::Instant;

mod common;

#[test]
#[ignore]
fn lysozyme_basis_matches_direct_solve() {
    let dir = common::pygbe_lysozyme_dir();
    let (vertices, faces) = read_msms(dir.join("Lys1.vert"), dir.join("Lys1.face")).unwrap();
    let surface = Surface::from_mesh(&vertices, &faces).unwrap();
    let charge_pqr: PathBuf =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pygbe_lys/built_parse.pqr");
    let atoms = read_pqr(&charge_pqr).unwrap();
    let media = Dielectric::continuum_with_salt(4.0, 80.0, 0.125);

    // why: five atom positions spread across the lysozyme structure;
    // any interior subset of `atoms.positions` works, but sampling
    // by index spaces them out rather than clustering on one
    // residue.
    let stride = atoms.positions.len() / 5;
    let sites: Vec<[f64; 3]> = (0..5).map(|k| atoms.positions[k * stride]).collect();
    let q = [0.7_f64, -0.4, 0.5, -0.9, 0.3];

    let t = Instant::now();
    let basis = LinearResponse::precompute(&surface, media, ChargeSide::Interior, &sites).unwrap();
    let elapsed = t.elapsed().as_secs_f64();
    eprintln!(
        "LinearResponse::precompute: {} sites, {} faces, elapsed {:.1}s",
        sites.len(),
        surface.num_faces(),
        elapsed,
    );

    let basis_energy = basis.solvation_energy(&q).unwrap();
    let direct = BemSolution::solve(&surface, media, ChargeSide::Interior, &sites, &q).unwrap();
    let mut direct_energy = 0.0;
    for (j, &_v) in q.iter().enumerate() {
        direct_energy += direct.interaction_energy(&sites, &q, 0, j).unwrap();
    }
    direct_energy *= 0.5;

    let rel = (basis_energy - direct_energy).abs() / direct_energy.abs();
    eprintln!(
        "E_solv: basis = {basis_energy:+.6e}, direct = {direct_energy:+.6e}, rel = {rel:.3e}"
    );
    // why: same tolerance the unit-level bilinear-consistency gate
    // uses — GMRES's `RELATIVE_TOL = 1e-5` per solve propagates into
    // an O(N_sites × 1e-5) reconstruction residual for the bilinear
    // composition.
    assert!(rel < 1e-4, "lysozyme basis vs direct: rel = {rel:.3e}");

    // Cross-check the batched site-potential path against the
    // general quadrature path on the same 5 sites: the G-based fast
    // path (symmetrised) and the raw reaction_field_at_many
    // (non-symmetric) should agree to the same discretisation scale.
    let mut via_g = vec![0.0; 5];
    let mut via_quad = vec![0.0; 5];
    basis.reaction_field_at_sites(&q, &mut via_g).unwrap();
    basis
        .reaction_field_at_many(&q, &sites, &mut via_quad)
        .unwrap();
    for (j, (&a, &b)) in via_g.iter().zip(&via_quad).enumerate() {
        let r = (a - b).abs() / a.abs().max(b.abs()).max(1e-12);
        assert!(r < 1e-3, "site {j}: via_g = {a}, via_quad = {b}, rel = {r}");
    }
}
