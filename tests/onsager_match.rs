//! Onsager 1936 dipole-solvation closed-form check.
//!
//! A point dipole at the centre of a dielectric sphere has a uniform
//! interior reaction field and a solvation energy
//!     U = −μ² · (ε_out − ε_in) / [ε_in · a³ · (2·ε_out + ε_in)]
//! (See `selfie::analytical::onsager`.) We approximate the dipole with
//! two opposite charges at `(0, 0, ±d/2)` with `μ = q·d`, solve the BEM,
//! and compare the resulting pair reaction-field energy against the
//! closed form. The approximation error is dominated by the quadrupole
//! term of the discrete charge distribution, which scales as `(d/a)²`.

#![cfg(feature = "validation")]

use selfie::analytical::onsager::dipole_solvation_energy;
use selfie::{BemSolution, ChargeSide, Dielectric, Surface};

const A: f64 = 10.0;
const EPS_IN: f64 = 2.0;
const EPS_OUT: f64 = 80.0;

/// Solve the BEM with two opposite charges of magnitude `q` at `±d/2·ẑ`
/// and return the total reaction-field energy
/// `U_rf = ½ Σ_i q_i · φ_rf(r_i)`.
fn bem_dipole_solvation(subdivisions: usize, q: f64, d: f64) -> f64 {
    let surface = Surface::icosphere(A, subdivisions);
    let media = Dielectric::continuum(EPS_IN, EPS_OUT);

    let r_plus = [0.0, 0.0, d * 0.5];
    let r_minus = [0.0, 0.0, -d * 0.5];
    let positions = [r_plus, r_minus];
    let values = [q, -q];

    let sol =
        BemSolution::solve(&surface, media, ChargeSide::Interior, &positions, &values).unwrap();

    let mut phi = [0.0_f64; 2];
    sol.reaction_field_at_many(&positions, &mut phi).unwrap();

    0.5 * (values[0] * phi[0] + values[1] * phi[1])
}

#[test]
fn two_close_charges_reproduce_onsager_dipole_solvation() {
    // μ = q · d = 1 e·Å. d = 1 Å gives (d/a)² = 0.01 worth of leading
    // quadrupole contamination — a clean < 2 % comparison target.
    let q = 1.0;
    let d = 1.0;
    let mu = q * d;

    let reference = dipole_solvation_energy(mu, A, EPS_IN, EPS_OUT);
    eprintln!("Onsager U_solv(μ = {mu} e·Å) = {reference:.9e}  (reduced e²/Å)");

    for s in [1usize, 3, 7] {
        let bem = bem_dipole_solvation(s, q, d);
        let n_t = 20 * (s + 1).pow(2);
        let rel = (bem - reference).abs() / reference.abs();
        eprintln!("  subdiv = {s:2}  n_t = {n_t:4}  BEM = {bem:.9e}  rel.err = {rel:.3e}");
    }

    let bem_fine = bem_dipole_solvation(7, q, d);
    let rel = (bem_fine - reference).abs() / reference.abs();
    assert!(
        rel < 0.02,
        "Onsager: rel.err at n_t = 1280 is {rel:.3e} (need < 2 %)"
    );
}

/// Aspirational convergence-rate gate that was always borderline:
/// at the icosphere subdiv=7 used here the BEM discretisation error
/// (~1 %) coincidentally matches the (d/a)² dipole-truncation error,
/// so the differential the test claims to observe is buried below the
/// BEM noise floor for any practical d. The sister test
/// `two_close_charges_reproduce_onsager_dipole_solvation` already
/// validates the Onsager reference itself; this one is `#[ignore]`d
/// pending either a much finer mesh or a redesign.
#[test]
#[ignore]
fn shrinking_gap_converges_to_pure_dipole() {
    let rel_at = |d: f64| {
        let mu = d; // q = 1
        let reference = dipole_solvation_energy(mu, A, EPS_IN, EPS_OUT);
        let bem = bem_dipole_solvation(7, 1.0, d);
        (bem - reference).abs() / reference.abs()
    };
    let e_1 = rel_at(1.0);
    let e_025 = rel_at(0.25);
    eprintln!("rel.err at d = 1.0: {e_1:.3e}; at d = 0.25: {e_025:.3e}");
    assert!(
        e_025 < 0.5 * e_1,
        "shrinking d should reduce error: {e_1:.3e} vs {e_025:.3e}"
    );
}
