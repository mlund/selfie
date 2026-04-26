//! MSMS `.vert` / `.face` pair reader (pygbe / NanoShaper convention).
//!
//! `.vert`: one vertex per whitespace-separated line. First three columns
//! are `x y z` (f64). Trailing columns (normals, atom labels) are
//! ignored. No header.
//!
//! `.face`: one face per line, three **1-indexed** integer vertex refs.
//! Trailing columns (region / property flags) ignored. Subtract 1 on load.
//!
//! Winding is CCW viewed from Ω⁺ (outward) by MSMS convention, matching
//! our `Surface` invariant — no reorientation is needed at load time.

use super::{Mesh, io_err, open, parse_f64};
use crate::error::Result;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Read an MSMS `.vert` + `.face` file pair.
///
/// # Errors
/// Returns [`Error::Io`] if either file can't be opened, a vertex line
/// has fewer than three numeric tokens, a face line has fewer than three
/// integer tokens, or a face references an out-of-range vertex index.
pub fn read_msms(vert_path: impl AsRef<Path>, face_path: impl AsRef<Path>) -> Result<Mesh> {
    let raw_vertices = read_vert(vert_path.as_ref())?;
    let raw_faces = read_face(face_path.as_ref(), raw_vertices.len())?;
    let (vertices, faces) = dedup_vertices(raw_vertices, raw_faces);
    // why: MSMS output occasionally contains a degenerate triangle
    // whose three vertex indices don't name three distinct points
    // — most commonly when two vertex rows in `.vert` have
    // bit-identical coordinates (seen on the pygbe lysozyme Lys8
    // mesh at one near-cone point) and the dedup pass collapses
    // them, leaving a face that references the same physical
    // vertex twice. Filtering those out here yields the same
    // topology pygbe uses, since a zero-area triangle contributes
    // nothing to the BEM integrals anyway.
    let n_before = faces.len();
    let faces: Vec<[u32; 3]> = faces
        .into_iter()
        .filter(|[a, b, c]| a != b && b != c && a != c)
        .collect();
    if faces.len() < n_before {
        log::info!(
            "MSMS mesh: dropped {} degenerate face(s) with coincident vertex indices",
            n_before - faces.len()
        );
    }
    log::info!(
        "MSMS mesh loaded: {} vertices, {} faces",
        vertices.len(),
        faces.len()
    );
    Ok((vertices, faces))
}

/// pygbe-style MSMS `.vert` files often store three vertices per face with
/// duplicates (triangle-soup). De-duplicate on exact bit equality so we
/// recover topological adjacency; face indices are remapped in place.
/// Vertices that differ only in f64 rounding are treated as distinct —
/// appropriate for meshes produced by exact-arithmetic subdivision; for
/// tool-generated meshes with small perturbations this can leave some
/// seams, but MSMS/NanoShaper output is bit-exact.
fn dedup_vertices(
    raw_vertices: Vec<[f64; 3]>,
    raw_faces: Vec<[u32; 3]>,
) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    use std::collections::HashMap;

    let mut unique: Vec<[f64; 3]> = Vec::with_capacity(raw_vertices.len());
    let mut remap: Vec<u32> = Vec::with_capacity(raw_vertices.len());
    let mut seen: HashMap<[u64; 3], u32> = HashMap::new();
    for v in &raw_vertices {
        let key = [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()];
        let new_idx = *seen.entry(key).or_insert_with(|| {
            let i = u32::try_from(unique.len()).expect("> u32::MAX unique vertices");
            unique.push(*v);
            i
        });
        remap.push(new_idx);
    }
    if unique.len() < raw_vertices.len() {
        log::info!(
            "MSMS .vert was triangle-soup: collapsed {} → {} unique vertices",
            raw_vertices.len(),
            unique.len()
        );
    }
    let faces = raw_faces
        .into_iter()
        .map(|[a, b, c]| [remap[a as usize], remap[b as usize], remap[c as usize]])
        .collect();
    (unique, faces)
}

fn read_vert(path: &Path) -> Result<Vec<[f64; 3]>> {
    let file = open(path)?;
    let mut out = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| io_err(path, line_no + 1, e.to_string()))?;
        let mut tokens = line.split_whitespace();
        let Some(tx) = tokens.next() else { continue };
        let ty = tokens
            .next()
            .ok_or_else(|| io_err(path, line_no + 1, "vertex line has fewer than 3 columns"))?;
        let tz = tokens
            .next()
            .ok_or_else(|| io_err(path, line_no + 1, "vertex line has fewer than 3 columns"))?;
        let x = parse_f64(path, line_no + 1, tx)?;
        let y = parse_f64(path, line_no + 1, ty)?;
        let z = parse_f64(path, line_no + 1, tz)?;
        out.push([x, y, z]);
    }
    Ok(out)
}

fn read_face(path: &Path, n_vertices: usize) -> Result<Vec<[u32; 3]>> {
    let file = open(path)?;
    let mut out = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| io_err(path, line_no + 1, e.to_string()))?;
        let mut tokens = line.split_whitespace();
        let Some(ta) = tokens.next() else { continue };
        let tb = tokens
            .next()
            .ok_or_else(|| io_err(path, line_no + 1, "face line has fewer than 3 columns"))?;
        let tc = tokens
            .next()
            .ok_or_else(|| io_err(path, line_no + 1, "face line has fewer than 3 columns"))?;
        let a = parse_one_indexed(path, line_no + 1, ta, n_vertices)?;
        let b = parse_one_indexed(path, line_no + 1, tb, n_vertices)?;
        let c = parse_one_indexed(path, line_no + 1, tc, n_vertices)?;
        out.push([a, b, c]);
    }
    Ok(out)
}

fn parse_one_indexed(path: &Path, line: usize, token: &str, n_vertices: usize) -> Result<u32> {
    let i: usize = token
        .parse()
        .map_err(|e| io_err(path, line, format!("expected usize, got {token:?}: {e}")))?;
    if i == 0 {
        return Err(io_err(
            path,
            line,
            "face index 0 is invalid (MSMS uses 1-indexed)",
        ));
    }
    if i > n_vertices {
        return Err(io_err(
            path,
            line,
            format!("face index {i} > vertex count {n_vertices}"),
        ));
    }
    Ok((i - 1) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(dir: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn parses_minimal_tetrahedron() {
        let dir = std::env::temp_dir().join("bemtzmann_msms_test_a");
        std::fs::create_dir_all(&dir).unwrap();
        let vert = write_tmp(
            &dir,
            "t.vert",
            "0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n",
        );
        let face = write_tmp(&dir, "t.face", "1 2 3\n1 4 2\n1 3 4\n2 4 3\n");
        let (verts, faces) = read_msms(&vert, &face).unwrap();
        assert_eq!(verts.len(), 4);
        assert_eq!(faces.len(), 4);
        assert_eq!(verts[0], [0.0, 0.0, 0.0]);
        assert_eq!(faces[0], [0, 1, 2]); // 1-indexed → 0-indexed
    }

    #[test]
    fn ignores_trailing_columns() {
        let dir = std::env::temp_dir().join("bemtzmann_msms_test_b");
        std::fs::create_dir_all(&dir).unwrap();
        let vert = write_tmp(
            &dir,
            "t.vert",
            "0.0 0.0 0.0 0.577 0.577 0.577 1 42 hello\n1.0 0.0 0.0 1.0 0.0 0.0 2 7 world\n\
             0.0 1.0 0.0 0.0 1.0 0.0 3 8 extra\n",
        );
        let face = write_tmp(&dir, "t.face", "1 2 3 region=0 label=xx\n");
        let (verts, faces) = read_msms(&vert, &face).unwrap();
        assert_eq!(verts[0], [0.0, 0.0, 0.0]);
        assert_eq!(faces[0], [0, 1, 2]);
    }

    #[test]
    fn reads_pygbe_sphere_geometry() {
        let root =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pygbe_sphere");
        let vert = root.join("sphere500_R4.vert");
        let face = root.join("sphere500_R4.face");
        let (verts, faces) = read_msms(vert, face).unwrap();
        // sphere500_R4: the "500" in the filename is approximate — the
        // actual mesh as shipped has 512 faces / 258 vertices.
        assert!(!faces.is_empty());
        assert!(!verts.is_empty());
        // Euler on a closed triangulation: V − E + F = 2, 2E = 3F
        // ⇒ V = F/2 + 2.
        assert_eq!(verts.len(), faces.len() / 2 + 2);
        // Every vertex lies on the sphere of radius 4 Å.
        for v in &verts {
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((r - 4.0).abs() < 1e-6, "vertex off sphere: r = {r}");
        }
    }

    #[test]
    fn rejects_face_index_out_of_range() {
        let dir = std::env::temp_dir().join("bemtzmann_msms_test_c");
        std::fs::create_dir_all(&dir).unwrap();
        let vert = write_tmp(&dir, "t.vert", "0 0 0\n1 0 0\n0 1 0\n");
        let face = write_tmp(&dir, "t.face", "1 2 9\n");
        assert!(read_msms(&vert, &face).is_err());
    }
}
