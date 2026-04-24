//! Shared geometry and dielectric constants for the sphere-Kirkwood
//! integration tests. Pair scale is salt-bridge-like (≈ 6.7 Å apart), both
//! charges 3 Å above the dielectric boundary.

#![cfg(feature = "validation")]
#![allow(dead_code)] // each test binary only uses a subset.

pub const A: f64 = 10.0; // sphere radius (Å)
pub const R: f64 = 13.0; // charge/eval radius (Å) — 3 Å standoff.
pub const EPS_IN: f64 = 2.0;
pub const EPS_OUT: f64 = 80.0;

/// Source point at `[R, 0, 0]`.
pub fn source() -> [f64; 3] {
    [R, 0.0, 0.0]
}

/// Evaluation point at γ = 30° from the source on the same |r| = R circle.
/// Chord length `2 R sin(15°) ≈ 6.73 Å`.
pub fn eval_point() -> [f64; 3] {
    [R * (3f64.sqrt() / 2.0), R * 0.5, 0.0]
}

// -----------------------------------------------------------------------
// pygbe regression-mesh fixtures — download on demand, cache, reuse.
// -----------------------------------------------------------------------

use std::path::PathBuf;

const PYGBE_MESH_ARCHIVE_URL: &str =
    "https://zenodo.org/record/55349/files/pygbe_regresion_test_meshes.zip";

/// Absolute path to the extracted pygbe lysozyme mesh directory,
/// downloaded on first use to a persistent cache.
///
/// The archive (`~9 MB`, zenodo.org) expands to `regresion_tests_meshes/Lysozyme/`
/// containing `Lys{1,2,4,8}.{vert,face}` plus stern/cavity variants we
/// don't use. We cache the extracted directory under
/// `$SELFIE_PYGBE_MESH_DIR` if set, else `$HOME/.cache/selfie/pygbe_meshes/`,
/// else a `target/` subdirectory. Subsequent test runs reuse the cache.
///
/// # Panics
/// On first use, if the download, unzip, or cache-write fails. Tests
/// that want to degrade gracefully should use [`pygbe_mesh_path_if_available`].
pub fn pygbe_lysozyme_dir() -> PathBuf {
    ensure_pygbe_archive_extracted().join("regresion_tests_meshes/Lysozyme")
}

/// Same as [`pygbe_lysozyme_dir`] but returns `None` instead of panicking
/// if the download fails (e.g. offline CI).
pub fn pygbe_mesh_path_if_available(filename: &str) -> Option<PathBuf> {
    match try_ensure_pygbe_archive_extracted() {
        Ok(dir) => Some(dir.join("regresion_tests_meshes/Lysozyme").join(filename)),
        Err(_) => None,
    }
}

fn cache_root() -> PathBuf {
    if let Ok(override_path) = std::env::var("SELFIE_PYGBE_MESH_DIR") {
        return PathBuf::from(override_path);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("selfie").join("pygbe_meshes");
    }
    // Last-resort fallback: inside the Cargo target dir. Survives cargo
    // build but not `cargo clean`.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("pygbe_meshes_cache")
}

fn ensure_pygbe_archive_extracted() -> PathBuf {
    try_ensure_pygbe_archive_extracted().expect("failed to fetch pygbe mesh archive")
}

fn try_ensure_pygbe_archive_extracted() -> Result<PathBuf, String> {
    let cache = cache_root();
    let marker = cache.join("regresion_tests_meshes/Lysozyme/Lys1.face");
    if marker.exists() {
        return Ok(cache);
    }
    std::fs::create_dir_all(&cache).map_err(|e| format!("mkdir {cache:?}: {e}"))?;
    let zip_path = cache.join("pygbe_meshes.zip");
    if !zip_path.exists() {
        let status = std::process::Command::new("curl")
            .args(["-sSL", "-o"])
            .arg(&zip_path)
            .arg(PYGBE_MESH_ARCHIVE_URL)
            .status()
            .map_err(|e| format!("curl invocation failed: {e}"))?;
        if !status.success() {
            return Err(format!("curl exited with {status}"));
        }
    }
    let status = std::process::Command::new("unzip")
        .args(["-qo"])
        .arg(&zip_path)
        .args(["-d"])
        .arg(&cache)
        .status()
        .map_err(|e| format!("unzip invocation failed: {e}"))?;
    if !status.success() {
        return Err(format!("unzip exited with {status}"));
    }
    if !marker.exists() {
        return Err(format!("archive did not contain expected marker {marker:?}"));
    }
    Ok(cache)
}
