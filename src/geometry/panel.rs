//! Per-face geometry computation.
//!
//! Kept struct-of-arrays (parallel `Vec<DVec3>` / `Vec<f64>`) rather than
//! array-of-structs so the public API can hand out contiguous `(N, 3)` and
//! `(N,)` slices to numpy via `repr(C)` transmute without allocating.

use glam::DVec3;

#[derive(Debug)]
pub struct FaceGeoms {
    pub centroids: Vec<DVec3>,
    pub normals: Vec<DVec3>,
    pub areas: Vec<f64>,
    // why: caching the three vertex `DVec3`s per face avoids an O(n²)
    // indirect-index + f32→f64 conversion in the assembly double loop and
    // an O(m·n) version in `reaction_field_at_many`. Costs 72 bytes/face
    // (well under 400 KB at n = 5120) and eliminates all per-pair vertex
    // lookups.
    pub tris: Vec<[DVec3; 3]>,
}

impl FaceGeoms {
    pub fn compute(vertices: &[DVec3], faces: &[[u32; 3]]) -> Self {
        let n = faces.len();
        let mut centroids = Vec::with_capacity(n);
        let mut normals = Vec::with_capacity(n);
        let mut areas = Vec::with_capacity(n);
        let mut tris = Vec::with_capacity(n);

        for &[ia, ib, ic] in faces {
            let a = vertices[ia as usize];
            let b = vertices[ib as usize];
            let c = vertices[ic as usize];

            // why: cross product magnitude = 2·area, direction = outward
            // normal for CCW winding viewed from outside. normalize_or_zero
            // avoids NaN on degenerate triangles; caller rejects those.
            let cross = (b - a).cross(c - a);
            centroids.push((a + b + c) / 3.0);
            normals.push(cross.normalize_or_zero());
            areas.push(0.5 * cross.length());
            tris.push([a, b, c]);
        }
        Self {
            centroids,
            normals,
            areas,
            tris,
        }
    }

    pub const fn len(&self) -> usize {
        self.areas.len()
    }

    #[allow(dead_code)]
    pub const fn is_empty(&self) -> bool {
        self.areas.is_empty()
    }
}
