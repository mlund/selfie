//! PQR charge-file reader (PDB with charge + radius in place of occupancy /
//! B-factor).
//!
//! Accepts both `ATOM` and `HETATM` records. Whitespace-tokenises each
//! record line and picks the last five numeric tokens as
//! `x y z charge radius`. The radius column feeds
//! [`crate::Surface::from_atoms_gaussian`] when meshing from atoms.

use super::{Atoms, io_err, open, parse_f64};
use crate::error::Result;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Read positions, charges, and radii from a PQR file.
///
/// # Errors
/// Returns [`Error::Io`] if the file can't be opened or an `ATOM` /
/// `HETATM` line has fewer than six whitespace tokens or any of the
/// trailing five fail to parse as `f64`.
pub fn read_pqr(path: impl AsRef<Path>) -> Result<Atoms> {
    let path = path.as_ref();
    let file = open(path)?;

    let mut positions = Vec::new();
    let mut charges = Vec::new();
    let mut radii = Vec::new();
    let mut skipped = 0_usize;

    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| io_err(path, line_no + 1, e.to_string()))?;
        let trimmed = line.trim_start();
        if !(trimmed.starts_with("ATOM") || trimmed.starts_with("HETATM")) {
            if !trimmed.is_empty() {
                skipped += 1;
            }
            continue;
        }
        // Anchor on the trailing numeric run (x, y, z, charge, radius):
        // the PDB/PQR residue-name column can be 3 or 4 chars wide, so
        // leading field positions drift, but the last five tokens don't.
        let (x, y, z, q, r) = parse_trailing_xyz_q_r(&line, path, line_no + 1)?;
        positions.push([x, y, z]);
        charges.push(q);
        radii.push(r);
    }

    if skipped > 0 {
        log::warn!(
            "PQR {}: skipped {} non-ATOM lines (REMARK/TER/END/etc.)",
            path.display(),
            skipped
        );
    }
    log::info!("PQR {}: {} atoms loaded", path.display(), charges.len());
    Ok(Atoms {
        positions,
        charges,
        radii,
    })
}

/// Extract (x, y, z, charge, radius) from the tail of a whitespace-tokenised
/// line without materialising a full `Vec<&str>`. Walks the iterator once,
/// keeping a sliding window of the last five tokens plus a total count.
fn parse_trailing_xyz_q_r(
    line: &str,
    path: &Path,
    line_no: usize,
) -> Result<(f64, f64, f64, f64, f64)> {
    let mut window: [&str; 5] = [""; 5];
    let mut total = 0_usize;
    for tok in line.split_whitespace() {
        window.rotate_left(1);
        window[4] = tok;
        total += 1;
    }
    if total < 6 {
        return Err(io_err(
            path,
            line_no,
            format!("ATOM/HETATM record has only {total} tokens, need ≥ 6"),
        ));
    }
    let x = parse_f64(path, line_no, window[0])?;
    let y = parse_f64(path, line_no, window[1])?;
    let z = parse_f64(path, line_no, window[2])?;
    let q = parse_f64(path, line_no, window[3])?;
    let r = parse_f64(path, line_no, window[4])?;
    Ok((x, y, z, q, r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(name: &str, content: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("bemtzmann_pqr_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn parses_minimal_pqr() {
        let path = write_tmp(
            "a.pqr",
            "REMARK header\nATOM 1 H HIS 5 1.0 2.0 3.0 -0.1 1.2\nEND\n",
        );
        let a = read_pqr(&path).unwrap();
        assert_eq!(a.positions, vec![[1.0, 2.0, 3.0]]);
        assert_eq!(a.charges, vec![-0.1]);
        assert_eq!(a.radii, vec![1.2]);
    }

    #[test]
    fn accepts_hetatm_records() {
        let path = write_tmp("b.pqr", "HETATM 1 ZN ZN 9999 0.5 0.5 0.5 2.0 1.4\n");
        let a = read_pqr(&path).unwrap();
        assert_eq!(a.positions, vec![[0.5, 0.5, 0.5]]);
        assert_eq!(a.charges, vec![2.0]);
        assert_eq!(a.radii, vec![1.4]);
    }

    #[test]
    fn reads_pygbe_offcenter() {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/pygbe_sphere/offcenter_R2.pqr");
        let a = read_pqr(&p).unwrap();
        assert_eq!(a.charges.len(), 1);
        assert!((a.charges[0] - 1.0).abs() < 1e-10);
        let p0 = a.positions[0];
        let r = (p0[0] * p0[0] + p0[1] * p0[1] + p0[2] * p0[2]).sqrt();
        assert!((r - 2.0).abs() < 1e-6);
        assert_eq!(a.radii.len(), 1);
        assert!(a.radii[0] > 0.0);
    }
}
