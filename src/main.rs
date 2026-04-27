//! `bemtzmann` command-line entry point.
//!
//! Pipeline: read a structure file → build a Gaussian molecular surface →
//! optionally write OBJ → optionally compute solvation free energy. The
//! whole binary is gated by the `cli` feature via `required-features` in
//! Cargo.toml; the library itself never links `clap`, `env_logger`, or
//! `indicatif`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use bemtzmann::io::{Atoms, read_pqr, read_xyz};
use bemtzmann::units::to_kJ_per_mol;
use bemtzmann::{BemSolution, ChargeSide, Dielectric, Error, Surface};
use clap::{Parser, ValueEnum};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use log::LevelFilter;

const KCAL_PER_KJ: f64 = 1.0 / 4.184;
// why: PQR charges are typically given to 3–4 decimals; anything
// smaller is rounding noise. Skipping these atoms saves O(N_panels)
// per-site work in both the GMRES RHS and the per-atom field
// evaluation without measurably perturbing E_solv.
const NEAR_ZERO_CHARGE: f64 = 1e-6;

/// BEMtzmann — boundary-element Poisson–Boltzmann CLI. Reads a structure
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
    #[arg(long, default_value_t = 1.5)]
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

    /// Sample the reaction-field potential on a 3D grid and write OpenDX
    /// (`.dx`) for visualisation in PyMOL or VMD. Implies `--solve`.
    #[arg(long, value_name = "PATH")]
    potential_dx: Option<PathBuf>,

    /// Grid spacing for `--potential-dx` in Å.
    #[arg(long, default_value_t = 1.0)]
    potential_spacing: f64,

    /// Padding around the atom bounding box for `--potential-dx`, in Å.
    #[arg(long, default_value_t = 5.0)]
    potential_padding: f64,

    /// Increase log verbosity (-v = debug, -vv = trace). Default: info.
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
    let multi = init_logger(cli.verbose);
    match run(&cli, &multi) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            if matches!(e, Error::MixedChargeSides { .. }) {
                eprintln!("hint: pass `--side interior` or `--side exterior` to override.");
            }
            ExitCode::FAILURE
        }
    }
}

/// Wires `env_logger` through `indicatif-log-bridge` so log records
/// suspend any active progress bar instead of tearing it. Returns the
/// shared [`MultiProgress`] every spinner must be added to.
fn init_logger(verbose: u8) -> MultiProgress {
    let level = match verbose {
        0 => LevelFilter::Info,
        1 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };
    let logger = env_logger::Builder::new()
        .filter_level(level)
        .format_target(false)
        .build();
    let multi = MultiProgress::new();
    // why: try_init can fail only if a global logger is already set,
    // which never happens in our binary; ignore the result and continue.
    let _ = LogWrapper::new(multi.clone(), logger).try_init();
    log::set_max_level(level);
    multi
}

/// Indeterminate spinner with a steady tick. Must be added to the shared
/// [`MultiProgress`] so log records (e.g. GMRES progress) coexist cleanly.
fn spinner(multi: &MultiProgress, msg: String) -> ProgressBar {
    let pb = multi.add(ProgressBar::new_spinner());
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {elapsed:>4} {msg}")
            .expect("static template literal is well-formed"),
    );
    pb.set_message(msg);
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Determinate progress bar for a known-length workload (e.g. grid points).
fn progress_bar(multi: &MultiProgress, total: u64, msg: String) -> ProgressBar {
    let pb = multi.add(ProgressBar::new(total));
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} {elapsed:>4} [{bar:30.cyan/blue}] {pos}/{len} ({percent}%) ETA {eta} {msg}",
        )
        .expect("static template literal is well-formed")
        .progress_chars("=> "),
    );
    pb.set_message(msg);
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

fn run(cli: &Cli, multi: &MultiProgress) -> Result<(), Error> {
    let atoms = load_atoms(&cli.structure, cli.format)?;
    log::info!(
        "Loaded {} atoms from {}",
        atoms.charges.len(),
        cli.structure.display()
    );

    let (charge_positions, charge_values) = filter_charged(&atoms);
    let dropped = atoms.charges.len() - charge_values.len();
    if dropped > 0 {
        log::info!(
            "Skipping {dropped} atoms with |q| < {NEAR_ZERO_CHARGE:.0e} for the solve ({} kept)",
            charge_values.len()
        );
    }

    let surface = {
        let pb = spinner(
            multi,
            format!(
                "Building Gaussian molecular surface (Δ = {:.2} Å)...",
                cli.grid_spacing
            ),
        );
        let t0 = Instant::now();
        let surface =
            Surface::from_atoms_gaussian(&atoms.positions, &atoms.radii, cli.grid_spacing)?;
        pb.finish_and_clear();
        log::debug!("Mesh built in {:.2?}", t0.elapsed());
        surface
    };

    let area: f64 = surface.face_areas().iter().sum();
    log::info!(
        "Mesh: {} vertices, {} faces, surface area {:.1} Å²",
        surface.num_vertices(),
        surface.num_faces(),
        area
    );

    if let Some(obj_path) = &cli.obj {
        surface.write_obj(obj_path)?;
        log::info!("Wrote {}", obj_path.display());
    }

    // why: `--potential-dx` needs a solved BemSolution too, so it
    // implicitly turns the solver on regardless of `--solve`.
    let needs_solve = cli.solve || cli.potential_dx.is_some();
    if needs_solve {
        let side = resolve_side(&surface, &charge_positions, cli.side)?;
        let media = Dielectric::continuum_with_salt(cli.eps_in, cli.eps_out, cli.kappa);
        log::info!(
            "Solving BEM system (ε_in = {:.2}, ε_out = {:.2}, κ = {:.3} Å⁻¹, {} panels)...",
            cli.eps_in,
            cli.eps_out,
            cli.kappa,
            surface.num_faces()
        );
        let solution = {
            let pb = spinner(multi, "Running GMRES...".to_string());
            let t0 = Instant::now();
            let solution =
                BemSolution::solve(&surface, media, side, &charge_positions, &charge_values)?;
            pb.finish_and_clear();
            log::debug!("Solver finished in {:.2?}", t0.elapsed());
            solution
        };

        if cli.solve {
            let e_reduced = solvation_energy_from(&solution, &charge_positions, &charge_values)?;
            let e_kj = to_kJ_per_mol(e_reduced);
            let e_kcal = e_kj * KCAL_PER_KJ;
            log::info!(
                "E_solv = {e_kj:+.2} kJ/mol  ({e_kcal:+.2} kcal/mol)  [{e_reduced:+.4} e²/Å]"
            );
        }

        if let Some(dx_path) = &cli.potential_dx {
            let (origin, spacing, dims) = derive_potential_grid(
                &atoms.positions,
                cli.potential_spacing,
                cli.potential_padding,
            );
            let t0 = Instant::now();
            sample_and_write_dx(multi, &solution, dx_path, origin, spacing, dims)?;
            log::debug!("Potential grid written in {:.2?}", t0.elapsed());
            log::info!(
                "Wrote {} (potential grid {}×{}×{}, spacing {:.2} Å)",
                dx_path.display(),
                dims[0],
                dims[1],
                dims[2],
                cli.potential_spacing
            );
        }
    }

    Ok(())
}

fn load_atoms(path: &Path, override_format: Option<InputFormat>) -> Result<Atoms, Error> {
    let format = override_format
        .or_else(|| detect_format(path))
        .ok_or_else(|| Error::Io {
            path: path.display().to_string(),
            reason: "could not infer format from extension; pass `--format pqr|xyz`".into(),
        })?;
    match format {
        InputFormat::Pqr => read_pqr(path),
        InputFormat::Xyz => read_xyz(path),
    }
}

fn detect_format(path: &Path) -> Option<InputFormat> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "pqr" => Some(InputFormat::Pqr),
        "xyz" => Some(InputFormat::Xyz),
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
    log::info!("Charge side: {} ({source})", side_label(side));
    Ok(side)
}

const fn side_label(side: ChargeSide) -> &'static str {
    match side {
        ChargeSide::Interior => "interior",
        ChargeSide::Exterior => "exterior",
    }
}

/// Split [`Atoms`] into the charged-site subset used by the solver.
/// Backbone / counterion atoms with `|q| < NEAR_ZERO_CHARGE` contribute
/// nothing meaningful to the RHS or to E_solv, so we skip the per-site
/// work for them. Radii are kept on the full [`Atoms`] for the mesher,
/// which still needs every atom to build the molecular surface.
fn filter_charged(atoms: &Atoms) -> (Vec<[f64; 3]>, Vec<f64>) {
    atoms
        .positions
        .iter()
        .copied()
        .zip(atoms.charges.iter().copied())
        .filter(|(_, q)| q.abs() >= NEAR_ZERO_CHARGE)
        .unzip()
}

/// Total solvation energy E_solv = ½ Σ qⱼ φ_rf(rⱼ) from an already-built
/// solution. Uses the batched `reaction_field_at_many` so the per-atom
/// field evaluations share rayon parallelism — O(N_panels) total instead
/// of O(N_atoms × N_panels) from looping `interaction_energy`.
fn solvation_energy_from(
    solution: &BemSolution<'_>,
    positions: &[[f64; 3]],
    charges: &[f64],
) -> Result<f64, Error> {
    let mut phi_rf = vec![0.0_f64; charges.len()];
    solution.reaction_field_at_many(positions, &mut phi_rf)?;
    let dot: f64 = charges.iter().zip(&phi_rf).map(|(&q, &p)| q * p).sum();
    Ok(0.5 * dot)
}

/// Sample `φ_rf` on a regular grid one x-slab at a time, writing the
/// result as an OpenDX (`.dx`) volume. Mirrors
/// `BemSolution::write_potential_dx`, but chunks the parallel sampling
/// per slab so the CLI can drive a determinate progress bar from
/// `n_total` known points. Library callers without a progress bar should
/// keep using `write_potential_dx`.
fn sample_and_write_dx(
    multi: &MultiProgress,
    solution: &BemSolution<'_>,
    path: &Path,
    origin: [f64; 3],
    spacing: [f64; 3],
    dims: [usize; 3],
) -> Result<(), Error> {
    use std::io::Write;
    let [nx, ny, nz] = dims;
    let n_total = nx * ny * nz;
    let slab_len = ny * nz;

    let io_err = |e: std::io::Error| Error::Io {
        path: path.display().to_string(),
        reason: e.to_string(),
    };

    // why: APBS / OpenDX storage order is x outermost, z innermost.
    // Sampling a full x-slab per call keeps rayon's per-point parallelism
    // while giving us nx synchronisation points to update the bar.
    let mut values = vec![0.0_f64; n_total];
    let mut slab_points: Vec<[f64; 3]> = Vec::with_capacity(slab_len);
    let pb = progress_bar(
        multi,
        n_total as u64,
        format!("Sampling reaction-field potential on {nx}×{ny}×{nz} grid"),
    );
    for i in 0..nx {
        slab_points.clear();
        let x = origin[0] + i as f64 * spacing[0];
        for j in 0..ny {
            let y = origin[1] + j as f64 * spacing[1];
            for k in 0..nz {
                let z = origin[2] + k as f64 * spacing[2];
                slab_points.push([x, y, z]);
            }
        }
        let slab_offset = i * slab_len;
        let slab_out = &mut values[slab_offset..slab_offset + slab_len];
        solution.reaction_field_at_many(&slab_points, slab_out)?;
        pb.inc(slab_len as u64);
    }
    pb.finish_and_clear();

    let file = std::fs::File::create(path).map_err(io_err)?;
    let mut w = std::io::BufWriter::new(file);
    writeln!(
        w,
        "# OpenDX scalar field — reaction-field potential φ_rf (reduced units, e/Å)"
    )
    .map_err(io_err)?;
    writeln!(w, "# Generated by BEMtzmann").map_err(io_err)?;
    writeln!(w, "object 1 class gridpositions counts {nx} {ny} {nz}").map_err(io_err)?;
    writeln!(w, "origin {} {} {}", origin[0], origin[1], origin[2]).map_err(io_err)?;
    writeln!(w, "delta {} 0 0", spacing[0]).map_err(io_err)?;
    writeln!(w, "delta 0 {} 0", spacing[1]).map_err(io_err)?;
    writeln!(w, "delta 0 0 {}", spacing[2]).map_err(io_err)?;
    writeln!(w, "object 2 class gridconnections counts {nx} {ny} {nz}").map_err(io_err)?;
    writeln!(
        w,
        "object 3 class array type double rank 0 items {n_total} data follows"
    )
    .map_err(io_err)?;
    for chunk in values.chunks(3) {
        for (i, v) in chunk.iter().enumerate() {
            if i > 0 {
                write!(w, " ").map_err(io_err)?;
            }
            write!(w, "{v:.7e}").map_err(io_err)?;
        }
        writeln!(w).map_err(io_err)?;
    }
    writeln!(w, "attribute \"dep\" string \"positions\"").map_err(io_err)?;
    writeln!(
        w,
        "object \"regular positions regular connections\" class field"
    )
    .map_err(io_err)?;
    writeln!(w, "component \"positions\" value 1").map_err(io_err)?;
    writeln!(w, "component \"connections\" value 2").map_err(io_err)?;
    writeln!(w, "component \"data\" value 3").map_err(io_err)?;
    w.flush().map_err(io_err)?;
    Ok(())
}

/// Build a regular cubic grid spanning the atom bounding box plus
/// `padding` on every side, with isotropic `spacing` Å steps. Returns
/// `(origin, [spacing; 3], dims)` ready to feed into the DX sampler.
fn derive_potential_grid(
    positions: &[[f64; 3]],
    spacing: f64,
    padding: f64,
) -> ([f64; 3], [f64; 3], [usize; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in positions {
        for ((l, h), &c) in lo.iter_mut().zip(hi.iter_mut()).zip(p) {
            *l = l.min(c);
            *h = h.max(c);
        }
    }
    let mut origin = [0.0_f64; 3];
    let mut dims = [0_usize; 3];
    for axis in 0..3 {
        origin[axis] = lo[axis] - padding;
        let upper = hi[axis] + padding;
        dims[axis] = ((upper - origin[axis]) / spacing).ceil() as usize + 1;
    }
    (origin, [spacing; 3], dims)
}
