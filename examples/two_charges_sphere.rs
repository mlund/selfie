//! Phase-1 milestone example: two opposite unit charges sitting 5 Å outside
//! a low-dielectric sphere in water (ε_in = 2, ε_out = 80, κ = 0). Prints the
//! reaction-field contribution to the pair interaction energy in reduced
//! units, kcal/mol and kT, and compares against the analytical Kirkwood
//! reference.

use selfie::units::{to_kcal_per_mol, to_kt};
use selfie::{BemSolution, Dielectric, Surface};

#[cfg(feature = "validation")]
use selfie::analytical::kirkwood::pair_reaction_energy;

fn main() {
    let a = 10.0_f64;
    let r = 13.0_f64; // 3 Å from sphere surface
    let subdivisions = 7;

    let surface = Surface::icosphere(a, subdivisions);
    let media = Dielectric::continuum(2.0, 80.0);
    // γ = 30° → chord 2 r sin(15°) ≈ 6.7 Å (salt-bridge pair).
    let positions = [[r, 0.0, 0.0], [r * (3.0_f64.sqrt() / 2.0), r * 0.5, 0.0]];
    let values = [1.0_f64, -1.0_f64];

    // Per-charge solve to isolate the pair reaction-field energy cleanly.
    // why: solving with *both* source charges mixes in each charge's Born
    // self-energy at its own location, which isn't part of the pairwise W_ij.
    let single_source = [positions[0]];
    let single_value = [values[0]];
    let sol = BemSolution::solve(&surface, media, &single_source, &single_value).unwrap();

    // φ_rf at r_2 from the q_1 solve; times q_2 gives W_12^rf.
    let phi_rf = sol.reaction_field_at(positions[1]);
    let w12_rf = values[1] * phi_rf;

    let chord = 2.0 * r * (15.0_f64.to_radians()).sin();
    println!(
        "Geometry: sphere a = {a} Å, charges at r = {r} Å (3 Å standoff), \
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
        "                      = {:+.4} kcal/mol",
        to_kcal_per_mol(w12_rf)
    );
    println!(
        "                      = {:+.4} kT (at 298.15 K)",
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
