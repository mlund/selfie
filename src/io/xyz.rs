//! Permissive whitespace columnar charge reader.
//!
//! One atom per line: `element x y z charge [radius]`. Element is a
//! string token (ignored in flight). Lines that don't match are
//! silently skipped — accepts classical XYZ headers (integer count on
//! line 1, free-form comment on line 2), extxyz `Properties=...`
//! metadata, blank lines, and `#`-comments without configuration.
//!
//! Skipped lines generate a `log::warn!` so they aren't invisible.

use super::{Charges, io_err, open};
use crate::error::Result;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Read charges from a whitespace columnar `.xyz` file.
///
/// # Errors
/// Returns [`Error::Io`] only if the file cannot be opened. Any other
/// malformed content results in a skipped line with a warning; the
/// caller should assert non-empty on the returned `Charges` if required.
pub fn read_xyz(path: impl AsRef<Path>) -> Result<Charges> {
    let path = path.as_ref();
    let file = open(path)?;

    let mut positions = Vec::new();
    let mut values = Vec::new();
    let mut skipped = 0_usize;

    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| io_err(path, line_no + 1, e.to_string()))?;
        match parse_atom_line(&line) {
            Some((x, y, z, q)) => {
                positions.push([x, y, z]);
                values.push(q);
            }
            None if !line.trim().is_empty() => skipped += 1,
            None => {}
        }
    }

    if skipped > 0 {
        log::warn!(
            "XYZ {}: skipped {} lines that didn't match `element x y z charge [radius]`",
            path.display(),
            skipped
        );
    }
    log::info!("XYZ {}: {} charges loaded", path.display(), values.len());
    Ok(Charges { positions, values })
}

/// Try to parse a single line as `element x y z charge [radius]`.
/// Returns `None` for anything that doesn't match the pattern so the
/// caller can transparently skip headers, comments, and extxyz metadata.
fn parse_atom_line(line: &str) -> Option<(f64, f64, f64, f64)> {
    if line.trim_start().starts_with('#') {
        return None;
    }
    // Single pass over tokens without allocating a Vec: peek the element,
    // then try to parse the next four as f64 for (x, y, z, charge); an
    // optional sixth (radius) must parse too if present.
    let mut tokens = line.split_whitespace();
    let element = tokens.next()?;
    // Element must not itself be numeric — otherwise a lone `3` (atom
    // count) would masquerade as element + 4 absent columns.
    if element.parse::<f64>().is_ok() {
        return None;
    }
    let x = tokens.next()?.parse::<f64>().ok()?;
    let y = tokens.next()?.parse::<f64>().ok()?;
    let z = tokens.next()?.parse::<f64>().ok()?;
    let q = tokens.next()?.parse::<f64>().ok()?;
    if let Some(radius) = tokens.next() {
        radius.parse::<f64>().ok()?;
        if tokens.next().is_some() {
            return None; // more than 6 tokens ⇒ reject
        }
    }
    Some((x, y, z, q))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(name: &str, content: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("selfie_xyz_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn parses_minimal_five_column() {
        let path = write_tmp("a.xyz", "C 0.0 0.0 0.0 -0.2\nH 1.0 0.0 0.0 +0.1\n");
        let c = read_xyz(&path).unwrap();
        assert_eq!(c.positions.len(), 2);
        assert_eq!(c.values, vec![-0.2, 0.1]);
    }

    #[test]
    fn accepts_optional_radius() {
        let path = write_tmp("b.xyz", "C 0.0 0.0 0.0 -0.2 1.7\n");
        let c = read_xyz(&path).unwrap();
        assert_eq!(c.values, vec![-0.2]);
    }

    #[test]
    fn silently_skips_xyz_header_and_comments() {
        let path = write_tmp(
            "c.xyz",
            "3\ncomment line — ignored\n# another comment\n\
             C 0.0 0.0 0.0 -0.2 1.7\n\
             H 0.5 0.5 0.5 +0.1\n\
             O 1.0 0.0 0.0 -0.5 1.52\n",
        );
        let c = read_xyz(&path).unwrap();
        assert_eq!(c.values.len(), 3);
    }

    #[test]
    fn ignores_extxyz_properties_header() {
        let path = write_tmp(
            "d.xyz",
            "3\n\
             Properties=species:S:1:pos:R:3:charge:R:1 Lattice=\"10 0 0 0 10 0 0 0 10\"\n\
             C 0.0 0.0 0.0 -0.2\n\
             H 0.5 0.5 0.5 +0.1\n\
             O 1.0 0.0 0.0 -0.5\n",
        );
        let c = read_xyz(&path).unwrap();
        assert_eq!(c.values.len(), 3);
    }

    #[test]
    fn returns_empty_when_no_atom_lines() {
        let path = write_tmp("e.xyz", "3\ncomment only, no data\n");
        let c = read_xyz(&path).unwrap();
        assert!(c.positions.is_empty());
    }
}
