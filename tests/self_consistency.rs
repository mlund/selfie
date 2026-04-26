//! BEM self-consistency checks that don't require any analytical reference.
//!
//! - **Reciprocity**: the Juffer operator is symmetric in
//!   `(r_i, r_j)` — the reaction-field potential at `r_j` due to a unit
//!   source at `r_i` must equal its swap. This is the sharpest check that
//!   the top/bottom-block signs, the ε_out/ε_in factor, and the evaluator
//!   orientation all agree.
//! - **No-contrast**: when ε_in = ε_out there is no dielectric boundary to
//!   polarise and every BEM reaction-field value must be numerically zero
//!   (bounded only by the quadrature and LU residuals).

#![cfg(feature = "validation")]

use bemtzmann::{BemSolution, ChargeSide, Dielectric, Surface};

const A: f64 = 10.0;
const EPS_IN: f64 = 2.0;
const EPS_OUT: f64 = 80.0;
const SUBDIV: usize = 7;

fn phi_rf_from_single_source(
    surface: &Surface,
    media: Dielectric,
    side: ChargeSide,
    source: [f64; 3],
    probe: [f64; 3],
) -> f64 {
    let sol = BemSolution::solve(surface, media, side, &[source], &[1.0]).unwrap();
    sol.reaction_field_at(probe)
}

fn assert_reciprocal(label: &str, side: ChargeSide, kappa: f64, r_i: [f64; 3], r_j: [f64; 3]) {
    let surface = Surface::icosphere(A, SUBDIV);
    let media = Dielectric::continuum_with_salt(EPS_IN, EPS_OUT, kappa);
    let phi_ij = phi_rf_from_single_source(&surface, media, side, r_i, r_j);
    let phi_ji = phi_rf_from_single_source(&surface, media, side, r_j, r_i);
    let rel = (phi_ij - phi_ji).abs() / phi_ij.abs().max(phi_ji.abs());
    eprintln!("{label}: φ(r_j;r_i) = {phi_ij:+.6e}, φ(r_i;r_j) = {phi_ji:+.6e}, rel = {rel:.3e}");
    // why: reciprocity of the continuous Green's function is exact, but the
    // *discretised* Juffer operator is only asymptotically symmetric —
    // centroid-collocation breaks strict matrix symmetry and the error
    // vanishes as O(h²). At n_t = 1280 the observed asymmetry is ~1e-4 for
    // both κ = 0 and κ > 0 geometries; we threshold at 1e-3 to catch gross
    // sign/scale bugs without flagging the honest discretisation residual.
    assert!(rel < 1e-3, "{label} reciprocity violated: rel = {rel:.3e}");
}

#[test]
fn reciprocity_interior_no_salt() {
    assert_reciprocal(
        "interior κ=0",
        ChargeSide::Interior,
        0.0,
        [7.0, 0.0, 0.0],
        [7.0 * (3f64.sqrt() / 2.0), 7.0 * 0.5, 0.0],
    );
}

#[test]
fn reciprocity_interior_with_salt() {
    // why: the only test that exercises off-centre interior charges with a
    // salty exterior — analytical references for this combination (general
    // Tanford–Kirkwood C_kl series) are not implemented, so reciprocity is
    // the self-consistency check guarding the production physics regime.
    assert_reciprocal(
        "interior κ=1/14",
        ChargeSide::Interior,
        1.0 / 14.0,
        [7.0, 0.0, 0.0],
        [7.0 * (3f64.sqrt() / 2.0), 7.0 * 0.5, 0.0],
    );
}

#[test]
fn reciprocity_exterior_no_salt() {
    assert_reciprocal(
        "exterior κ=0",
        ChargeSide::Exterior,
        0.0,
        [13.0, 0.0, 0.0],
        [13.0 * (3f64.sqrt() / 2.0), 13.0 * 0.5, 0.0],
    );
}

#[test]
fn reciprocity_exterior_with_salt() {
    assert_reciprocal(
        "exterior κ=1/14",
        ChargeSide::Exterior,
        1.0 / 14.0,
        [13.0, 0.0, 0.0],
        [13.0 * (3f64.sqrt() / 2.0), 13.0 * 0.5, 0.0],
    );
}

fn assert_no_contrast_zero(label: &str, side: ChargeSide, source: [f64; 3], probe: [f64; 3]) {
    let surface = Surface::icosphere(A, SUBDIV);
    let media = Dielectric::continuum(80.0, 80.0); // ε_in = ε_out ⇒ no boundary.
    let phi = phi_rf_from_single_source(&surface, media, side, source, probe);
    // why: for ε_in = ε_out the physical reaction field is exactly zero;
    // the BEM result is bounded by 3-point Gauss quadrature error, which
    // at n_t = 1280 stays comfortably below 1e-5 in reduced e/Å for the
    // chosen geometries (about 3 orders of magnitude below the contrast
    // results we compare against elsewhere).
    eprintln!("{label}: φ_rf = {phi:+.3e}");
    assert!(
        phi.abs() < 1e-5,
        "{label}: |φ_rf| = {:.3e} (expected ≈ 0)",
        phi.abs()
    );
}

#[test]
fn no_contrast_interior_probe() {
    assert_no_contrast_zero(
        "interior no-contrast",
        ChargeSide::Interior,
        [7.0, 0.0, 0.0],
        [7.0 * (3f64.sqrt() / 2.0), 7.0 * 0.5, 0.0],
    );
}

#[test]
fn no_contrast_exterior_probe() {
    assert_no_contrast_zero(
        "exterior no-contrast",
        ChargeSide::Exterior,
        [13.0, 0.0, 0.0],
        [13.0 * (3f64.sqrt() / 2.0), 13.0 * 0.5, 0.0],
    );
}
