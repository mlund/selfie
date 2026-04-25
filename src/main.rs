//! `selfie` command-line entry point.
//!
//! Pipeline: read a structure file → build a Gaussian molecular surface →
//! optionally write OBJ → optionally compute solvation free energy. The
//! whole binary is gated by the `cli` feature via `required-features` in
//! Cargo.toml; the library itself never links `clap` or `env_logger`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use log::LevelFilter;
use selfie::io::{Atoms, read_pqr, read_xyz};
use selfie::units::to_kJ_per_mol;
use selfie::{BemSolution, ChargeSide, Dielectric, Error, Surface};

const KCAL_PER_KJ: f64 = 1.0 / 4.184;

/// SelFie — boundary-element Poisson–Boltzmann CLI. Reads a structure
/// file, builds a Gaussian molecular surface, optionally writes the
/// mesh and/or computes the solvation free energy.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Atom file (.pqr or .xyz; format auto-detected by extension).
    structure: PathBuf,

    /// Override input-format detection.
    #[arg(long, value_enum)]
    format: Option<InputFormat>,

    /// Marching-cubes grid spacing in Å (smaller = finer mesh, quadratic memory).
    #[arg(long, default_value_t = 0.5)]
    grid_spacing: f64,

    /// Write the meshed surface as Wavefront OBJ.
    #[arg(long, value_name = "PATH")]
    obj: Option<PathBuf>,

    /// Compute the solvation free energy (interior charges by default).
    #[arg(long)]
    solve: bool,

    /// Interior dielectric constant.
    #[arg(long, default_value_t = 4.0)]
    eps_in: f64,

    /// Exterior dielectric constant.
    #[arg(long, default_value_t = 80.0)]
    eps_out: f64,

    /// Inverse Debye length, Å⁻¹ (0 disables salt screening).
    #[arg(long, default_value_t = 0.0)]
    kappa: f64,

    /// Override charge-side classification.
    #[arg(long, value_enum, default_value_t = SideArg::Auto)]
    side: SideArg,

    /// Increase log verbosity (-v = info, -vv = debug). Default: warnings only.
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum InputFormat {
    Pqr,
    Xyz,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum SideArg {
    Auto,
    Interior,
    Exterior,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    init_logger(cli.verbose);
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            if let Error::MixedChargeSides { .. } = e {
                eprintln!("hint: pass `--side interior` or `--side exterior` to override.");
            }
            ExitCode::FAILURE
        }
    }
}

fn init_logger(verbose: u8) {
    let level = match verbose {
        0 => LevelFilter::Warn,
        1 => LevelFilter::Info,
        _ => LevelFilter::Debug,
    };
    env_logger::Builder::new()
        .filter_level(level)
        .format_target(false)
        .init();
}

fn run(cli: &Cli) -> Result<(), Error> {
    let atoms = load_atoms(&cli.structure, cli.format)?;
    println!(
        "Loaded {} atoms from {}",
        atoms.charges.len(),
        cli.structure.display()
    );

    let surface = Surface::from_atoms_gaussian(&atoms.positions, &atoms.radii, cli.grid_spacing)?;
    let area: f64 = surface.face_areas().iter().sum();
    println!(
        "Mesh: {} vertices, {} faces, surface area {:.1} Å²",
        surface.num_vertices(),
        surface.num_faces(),
        area
    );

    if let Some(obj_path) = &cli.obj {
        surface.write_obj(obj_path)?;
        println!("Wrote {}", obj_path.display());
    }

    if cli.solve {
        let side = resolve_side(&surface, &atoms.positions, cli.side)?;
        let media = Dielectric::continuum_with_salt(cli.eps_in, cli.eps_out, cli.kappa);
        let e_reduced = compute_solvation_energy(&surface, media, side, &atoms)?;
        let e_kj = to_kJ_per_mol(e_reduced);
        let e_kcal = e_kj * KCAL_PER_KJ;
        println!(
            "E_solv = {e_kj:+.2} kJ/mol  ({e_kcal:+.2} kcal/mol)  [{e_reduced:+.4} e²/Å]"
        );
    }

    Ok(())
}

fn load_atoms(path: &Path, override_format: Option<InputFormat>) -> Result<Atoms, Error> {
    let format = override_format.or_else(|| detect_format(path)).ok_or_else(|| {
        Error::Io {
            path: path.display().to_string(),
            reason: "could not infer format from extension; pass `--format pqr|xyz`".into(),
        }
    })?;
    match format {
        InputFormat::Pqr => read_pqr(path),
        InputFormat::Xyz => read_xyz(path),
    }
}

fn detect_format(path: &Path) -> Option<InputFormat> {
    match path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase) {
        Some(ref ext) if ext == "pqr" => Some(InputFormat::Pqr),
        Some(ref ext) if ext == "xyz" => Some(InputFormat::Xyz),
        _ => None,
    }
}

fn resolve_side(
    surface: &Surface,
    positions: &[[f64; 3]],
    arg: SideArg,
) -> Result<ChargeSide, Error> {
    let (side, source) = match arg {
        SideArg::Interior => (ChargeSide::Interior, "forced"),
        SideArg::Exterior => (ChargeSide::Exterior, "forced"),
        SideArg::Auto => (surface.classify_charges(positions)?, "auto-classified"),
    };
    println!("Charge side: {} ({source})", side_label(side));
    Ok(side)
}

const fn side_label(side: ChargeSide) -> &'static str {
    match side {
        ChargeSide::Interior => "interior",
        ChargeSide::Exterior => "exterior",
    }
}

/// Total solvation energy E_solv = ½ Σ qⱼ φ_rf(rⱼ). Uses the batched
/// `reaction_field_at_many` so the per-atom field evaluations share rayon
/// parallelism — O(N_panels) total instead of O(N_atoms × N_panels) from
/// looping `interaction_energy`.
fn compute_solvation_energy(
    surface: &Surface,
    media: Dielectric,
    side: ChargeSide,
    atoms: &Atoms,
) -> Result<f64, Error> {
    let solution = BemSolution::solve(surface, media, side, &atoms.positions, &atoms.charges)?;
    let mut phi_rf = vec![0.0_f64; atoms.charges.len()];
    solution.reaction_field_at_many(&atoms.positions, &mut phi_rf)?;
    let dot: f64 = atoms.charges.iter().zip(&phi_rf).map(|(&q, &p)| q * p).sum();
    Ok(0.5 * dot)
}
