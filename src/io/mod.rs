//! File readers for mesh and atom data.
//!
//! Exactly four public items: [`read_msms`] for MSMS-format meshes
//! (`.vert` + `.face` pair — pygbe/NanoShaper convention), [`read_pqr`]
//! for PQR atom records, [`read_xyz`] for a permissive whitespace
//! columnar format (`element x y z charge [radius]`), and [`Atoms`]
//! as the return type for the two atom-record readers.
//!
//! All readers emit diagnostics via the `log` crate — `info!` for
//! successful counts and `warn!` for tolerated oddities (lines that
//! don't parse, unexpected trailing columns). Install an `env_logger`
//! or similar to see them; they're silent by default.

mod msms;
mod pqr;
mod xyz;

pub use msms::read_msms;
pub use pqr::read_pqr;
pub use xyz::read_xyz;

use crate::error::{Error, Result};
use std::fs::File;
use std::path::Path;

/// Parsed mesh as returned by [`read_msms`]. Pass to
/// [`crate::Surface::from_mesh`] as
/// `from_mesh(&mesh.vertices, &mesh.faces)`.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
}

/// Open a file for reading, mapping I/O errors into [`Error::Io`] with
/// the path embedded for diagnostics.
pub(in crate::io) fn open(path: &Path) -> Result<File> {
    File::open(path).map_err(|e| Error::Io {
        path: path.display().to_string(),
        reason: e.to_string(),
    })
}

/// Produce an [`Error::Io`] anchored at a specific line of a specific
/// file. All three readers call this from their parse paths.
pub(in crate::io) fn io_err(path: &Path, line: usize, msg: impl Into<String>) -> Error {
    Error::Io {
        path: path.display().to_string(),
        reason: format!("line {line}: {}", msg.into()),
    }
}

/// Parse a single token as `f64`, attaching a readable path+line context
/// if it fails.
pub(in crate::io) fn parse_f64(path: &Path, line: usize, token: &str) -> Result<f64> {
    token
        .parse::<f64>()
        .map_err(|e| io_err(path, line, format!("expected f64, got {token:?}: {e}")))
}

/// Atom records produced by [`read_pqr`] and [`read_xyz`].
///
/// Layout is the same SoA `&[[f64; 3]]` / `&[f64]` shape used by
/// [`crate::BemSolution::solve`], so the parsed data flows through
/// without repacking.
///
/// `radii` is populated when the source format carries a radius column —
/// always for PQR, optionally for XYZ. When the file has no radii the
/// field is an empty `Vec` (callers that need radii should check
/// `radii.len() == positions.len()` before using). The radii feed
/// directly into [`crate::Surface::from_atoms_gaussian`].
#[derive(Debug, Clone, Default)]
pub struct Atoms {
    pub positions: Vec<[f64; 3]>,
    pub charges: Vec<f64>,
    pub radii: Vec<f64>,
}
