//! File readers for mesh and charge data.
//!
//! Exactly four public items: [`read_msms`] for MSMS-format meshes
//! (`.vert` + `.face` pair — pygbe/NanoShaper convention), [`read_pqr`]
//! for PQR-format charges, [`read_xyz`] for a permissive whitespace
//! columnar format (`element x y z charge [radius]`), and [`Charges`]
//! as the return type for the two charge readers.
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
/// [`crate::Surface::from_mesh`] as `from_mesh(&mesh.0, &mesh.1)`.
pub type Mesh = (Vec<[f64; 3]>, Vec<[u32; 3]>);

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

/// Point charges produced by [`read_pqr`] and [`read_xyz`].
///
/// Layout is the same SoA `&[[f64; 3]]` / `&[f64]` shape used by
/// [`crate::BemSolution::solve`], so the parsed data flows through
/// without repacking.
#[derive(Debug, Clone, Default)]
pub struct Charges {
    pub positions: Vec<[f64; 3]>,
    pub values: Vec<f64>,
}
