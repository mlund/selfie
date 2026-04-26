//! Two unit charges inside a low-dielectric sphere in water (ε_in = 2,
//! ε_out = 80, κ = 0) — the canonical MC-pKa titration setup in miniature.
//! Prints the reaction-field contribution to the pair interaction energy
//! in reduced units, kJ/mol, and kT, and compares against the classical
//! Kirkwood 1934 interior-charge analytical reference.

use bemtzmann::units::{to_kJ_per_mol, to_kt};
use bemtzmann::{BemSolution, ChargeSide, Dielectric, Surface};

#[cfg(feature = "validation")]
use bemtzmann::analytical::kirkwood_inside::pair_reaction_energy;

fn main() {
    let a = 10.0_f64;
    let r = 7.0_f64; // 3 Å beneath the surface
    let subdivisions = 7;

    let surface = Surface::icosphere(a, subdivisions);
    let media = Dielectric::continuum(2.0, 80.0);
    // γ = 30° → chord 2 r sin(15°) ≈ 3.6 Å (buried salt-bridge pair).
    let positions = [[r, 0.0, 0.0], [r * (3.0_f64.sqrt() / 2.0), r * 0.5, 0.0]];
    let values = [1.0_f64, -1.0_f64];

    // Per-source solve so φ_rf(r_j) × q_j is the cross-term pair energy
    // without the source's own Born self-energy mixed in.
    let single_source = [positions[0]];
    let single_value = [values[0]];
    let sol = BemSolution::solve(
        &surface,
        media,
        ChargeSide::Interior,
        &single_source,
        &single_value,
    )
    .unwrap();
    let phi_rf = sol.reaction_field_at(positions[1]);
    let w12_rf = values[1] * phi_rf;

    let chord = 2.0 * r * (15.0_f64.to_radians()).sin();
    println!(
        "Geometry: sphere a = {a} Å, charges at r = {r} Å (3 Å beneath surface), \
         30° apart → pair distance {chord:.2} Å."
    );
    println!("Dielectric: ε_in = 2, ε_out = 80, κ = 0.");
    println!(
        "Mesh: hexasphere subdiv = {subdivisions} → {} triangles.",
        surface.num_faces()
    );
    println!();
    println!("W_12 (reaction field) = {w12_rf:+.6e} e²/Å");
    println!(
        "                      = {:+.3} kJ/mol",
        to_kJ_per_mol(w12_rf)
    );
    println!(
        "                      = {:+.3} kT (at 298.15 K)",
        to_kt(w12_rf, 298.15)
    );

    #[cfg(feature = "validation")]
    {
        let w_ref = pair_reaction_energy(
            positions[0],
            values[0],
            positions[1],
            values[1],
            a,
            2.0,
            80.0,
            80,
        );
        println!();
        println!("Kirkwood analytical W_12 (ref)= {w_ref:+.6e} e²/Å");
        let rel_err = (w12_rf - w_ref).abs() / w_ref.abs();
        println!("Relative error: {rel_err:.3e}");
    }
}
