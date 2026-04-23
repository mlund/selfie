//! Triangulated dielectric boundary Γ.
//!
//! The only public type is [`Surface`]. All panel geometry, mesh construction
//! details, and the `hexasphere` adapter are hidden behind it.

mod icosphere;
mod panel;

use crate::error::{Error, Result};
use crate::units::ChargeSide;
use glam::DVec3;
use panel::FaceGeoms;

/// A closed, outward-oriented triangulated boundary.
///
/// Invariants maintained after construction:
/// - Every face index is `< vertices.len()`.
/// - No face is degenerate (area > 0).
/// - Face normals point into Ω⁺ (outward). The constructors enforce this for
///   meshes enclosing the origin; [`Self::from_mesh`] returns
///   [`Error::NormalOrientation`] on failure.
#[derive(Debug)]
pub struct Surface {
    // why: glam's DVec3 is #[repr(C)] = three packed f64 (verified at
    // construction time by the const-asserts below), so `&[DVec3]` and
    // `&[[f64; 3]]` are layout-compatible. We use DVec3 internally for the
    // ergonomic `.dot`/`.cross`/`.normalize` math and expose the same bytes as
    // `&[[f64; 3]]` on the public API — zero-copy, numpy-compatible.
    vertices: Vec<DVec3>,
    faces: Vec<[u32; 3]>,
    geom: FaceGeoms,
}

// why: compile-time proof that the DVec3 → [f64; 3] reinterpret below is
// sound. If a future glam version changes this, the crate fails to compile
// rather than silently returning garbage at runtime.
const _: () = {
    assert!(core::mem::size_of::<DVec3>() == core::mem::size_of::<[f64; 3]>());
    assert!(core::mem::align_of::<DVec3>() <= core::mem::align_of::<[f64; 3]>());
};

impl Surface {
    /// Build an icosphere of radius `radius` using `hexasphere`'s subdivision.
    ///
    /// `subdivisions` is the number of **extra points per edge**
    /// (hexasphere convention — *not* a recursive-doubling level).
    /// Triangle count is `20 · (subdivisions + 1)²`; so `subdivisions = 7`
    /// gives 1280 triangles.
    #[must_use]
    pub fn icosphere(radius: f64, subdivisions: usize) -> Self {
        let (vertices, faces) = icosphere::build(radius, subdivisions);
        let geom = FaceGeoms::compute(&vertices, &faces);
        debug_assert!(
            geom.centroids
                .iter()
                .zip(geom.normals.iter())
                .all(|(c, n)| c.dot(*n) > 0.0),
            "icosphere produced inward-facing normal — hexasphere convention broke"
        );
        Self {
            vertices,
            faces,
            geom,
        }
    }

    /// Build a surface from caller-supplied vertex and face arrays.
    ///
    /// `vertices` layout: `&[[x, y, z], ...]`. `faces` layout: `&[[i, j, k], ...]`.
    /// Faces must be CCW as seen from Ω⁺ (outward). For protein surfaces this
    /// is the standard convention emitted by MSMS/NanoShaper.
    ///
    /// Sanity checks: non-empty, face indices in range, non-degenerate
    /// faces, outward normals (for meshes enclosing the origin), and the
    /// mesh is a closed orientable 2-manifold (every edge shared by
    /// exactly two faces with opposite half-edge orientations).
    ///
    /// # Errors
    /// Returns [`Error::MeshFaceOutOfRange`], [`Error::DegenerateFace`],
    /// [`Error::NormalOrientation`], or [`Error::NonManifoldMesh`] as
    /// appropriate.
    pub fn from_mesh(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> Result<Self> {
        if vertices.is_empty() || faces.is_empty() {
            return Err(Error::NonManifoldMesh {
                reason: format!(
                    "empty mesh: {} vertices, {} faces",
                    vertices.len(),
                    faces.len()
                ),
            });
        }
        let verts: Vec<DVec3> = vertices.iter().copied().map(DVec3::from).collect();
        let faces: Vec<[u32; 3]> = faces.to_vec();

        for (face_idx, &[a, b, c]) in faces.iter().enumerate() {
            for idx in [a, b, c] {
                if (idx as usize) >= verts.len() {
                    return Err(Error::MeshFaceOutOfRange {
                        face: face_idx,
                        index: idx,
                        vertex_count: verts.len(),
                    });
                }
            }
        }

        check_closed_orientable(&faces)?;

        let geom = FaceGeoms::compute(&verts, &faces);
        for (i, &area) in geom.areas.iter().enumerate() {
            if area <= 0.0 || !area.is_finite() {
                return Err(Error::DegenerateFace { face: i, area });
            }
        }

        // Divergence theorem: ∫_∂V x·n dS = 3V for a closed surface with
        // outward-pointing normals. The aggregate sign is translation-
        // invariant — a positive value for any closed-orientable mesh at
        // any position in space, negative only if the winding is inverted.
        // We use this instead of a per-face centroid·normal check since
        // the latter only works for origin-centred meshes.
        let signed_volume_6: f64 = geom
            .centroids
            .iter()
            .zip(&geom.normals)
            .zip(&geom.areas)
            .map(|((&c, &n), &a)| c.dot(n) * a)
            .sum();
        if signed_volume_6 <= 0.0 {
            return Err(Error::NormalOrientation {
                face: 0,
                dot: signed_volume_6,
            });
        }

        Ok(Self {
            vertices: verts,
            faces,
            geom,
        })
    }

    pub const fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub const fn num_faces(&self) -> usize {
        self.geom.len()
    }

    /// Vertex positions as a contiguous `(N, 3)` f64 slice.
    pub fn vertices(&self) -> &[[f64; 3]] {
        reinterpret_dvec3_slice(&self.vertices)
    }

    /// Face vertex indices as `(N, 3)` u32.
    pub fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }

    /// Centroid of each face as `(N, 3)` f64.
    pub fn face_centroids(&self) -> &[[f64; 3]] {
        reinterpret_dvec3_slice(&self.geom.centroids)
    }

    /// Outward unit normal of each face as `(N, 3)` f64.
    pub fn face_normals(&self) -> &[[f64; 3]] {
        reinterpret_dvec3_slice(&self.geom.normals)
    }

    /// Area of each face, length `num_faces`.
    pub fn face_areas(&self) -> &[f64] {
        &self.geom.areas
    }

    /// Infer whether a set of probe points all live inside or all outside
    /// the closed surface, via ray-casting along +x.
    ///
    /// Intended use: given charges loaded from a PQR or XYZ file (which
    /// carry no side information), pick the correct [`ChargeSide`] to pass
    /// to [`crate::BemSolution::solve`] without the user having to assert
    /// which side they're on.
    ///
    /// # Errors
    /// Returns [`Error::MixedChargeSides`] if the points don't all share a
    /// single side (some interior, some exterior, or any within tolerance
    /// of the mesh surface itself).
    pub fn classify_charges(&self, positions: &[[f64; 3]]) -> Result<ChargeSide> {
        let mut interior = 0_usize;
        let mut exterior = 0_usize;
        let mut on_surface = 0_usize;
        for &p in positions {
            match self.contains_point(DVec3::from(p)) {
                PointLocation::Interior => interior += 1,
                PointLocation::Exterior => exterior += 1,
                PointLocation::OnSurface => on_surface += 1,
            }
        }
        match (interior, exterior, on_surface) {
            (_, 0, 0) => Ok(ChargeSide::Interior),
            (0, _, 0) => Ok(ChargeSide::Exterior),
            _ => Err(Error::MixedChargeSides {
                interior,
                exterior,
                on_surface,
            }),
        }
    }

    /// Ray-cast `p` along a perturbed +x direction and count crossings with
    /// the mesh's triangles. Odd count → interior for a closed orientable
    /// surface.
    fn contains_point(&self, p: DVec3) -> PointLocation {
        // why: icospheres have symmetry axes aligned with +x; firing
        // exactly along +x from the origin (or any point on a symmetry
        // line) grazes triangle edges and the crossing parity becomes
        // meaningless. Irrational small perturbations move the ray off
        // every rational alignment while staying well inside the triangle
        // we'd have hit for a generic point.
        const EPS_EDGE: f64 = 1e-10;
        let mut crossings = 0_i32;
        for &[ia, ib, ic] in &self.faces {
            let face = [
                self.vertices[ia as usize],
                self.vertices[ib as usize],
                self.vertices[ic as usize],
            ];
            match ray_triangle_intersect(p, RAY_DIR, face, EPS_EDGE) {
                PointLocation::Interior => crossings += 1,
                PointLocation::Exterior => {}
                PointLocation::OnSurface => return PointLocation::OnSurface,
            }
        }
        if crossings & 1 == 1 {
            PointLocation::Interior
        } else {
            PointLocation::Exterior
        }
    }

    pub(crate) const fn geom_internal(&self) -> &FaceGeoms {
        &self.geom
    }
}

enum PointLocation {
    Interior,
    Exterior,
    OnSurface,
}

/// Irrational-offset ray direction for point-in-mesh testing. Kept as a
/// module-level `const` so the loop doesn't re-materialise it per face.
const RAY_DIR: DVec3 = DVec3::new(1.0, 1.234e-5, -2.718e-5);

/// Möller-Trumbore ray-triangle intersection.
///
/// Reused from `contains_point`: the three-way return (`Interior` for a
/// forward crossing, `Exterior` for miss / backwards, `OnSurface` when
/// the ray origin lies on the triangle plane within `eps`) keeps the
/// parity counter clean and lets a grazing point short-circuit the whole
/// mesh loop.
fn ray_triangle_intersect(origin: DVec3, dir: DVec3, face: [DVec3; 3], eps: f64) -> PointLocation {
    let e1 = face[1] - face[0];
    let e2 = face[2] - face[0];
    let h = dir.cross(e2);
    let det = e1.dot(h);
    if det.abs() < eps {
        return PointLocation::Exterior;
    }
    let inv_det = 1.0 / det;
    let s = origin - face[0];
    let u = s.dot(h) * inv_det;
    if !(-eps..=1.0 + eps).contains(&u) {
        return PointLocation::Exterior;
    }
    let q = s.cross(e1);
    let v = dir.dot(q) * inv_det;
    if v < -eps || u + v > 1.0 + eps {
        return PointLocation::Exterior;
    }
    let t = e2.dot(q) * inv_det;
    if t.abs() < eps {
        PointLocation::OnSurface
    } else if t > 0.0 {
        PointLocation::Interior
    } else {
        PointLocation::Exterior
    }
}

/// Verify that the mesh is a closed orientable 2-manifold: every directed
/// half-edge appears exactly once and its opposite also appears exactly
/// once. O(F) via a hash set, single pass — we track paired-so-far and
/// compare against the expected 3·F at the end.
fn check_closed_orientable(faces: &[[u32; 3]]) -> Result<()> {
    use std::collections::HashSet;

    let mut half_edges: HashSet<(u32, u32)> = HashSet::with_capacity(3 * faces.len());
    let mut paired = 0_usize;
    for (face_idx, &[a, b, c]) in faces.iter().enumerate() {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            if u == v {
                return Err(Error::NonManifoldMesh {
                    reason: format!("face {face_idx} has repeated vertex {u}"),
                });
            }
            if !half_edges.insert((u, v)) {
                return Err(Error::NonManifoldMesh {
                    reason: format!(
                        "directed edge ({u}, {v}) appears in more than one face \
                         — duplicate faces or inconsistent winding"
                    ),
                });
            }
            if half_edges.contains(&(v, u)) {
                paired += 1;
            }
        }
    }
    // why: valid closed mesh ⇔ every half-edge has its opposite ⇔ every
    // edge contributes a pair, so `paired == half_edges.len() / 2`.
    if paired * 2 != half_edges.len() {
        return Err(Error::NonManifoldMesh {
            reason: format!(
                "{} half-edges have no matching opposite: mesh is open or \
                 has inconsistent winding",
                half_edges.len() - paired * 2
            ),
        });
    }
    Ok(())
}

/// Safe reinterpretation of `&[DVec3]` as `&[[f64; 3]]` based on the const
/// layout asserts above.
const fn reinterpret_dvec3_slice(v: &[DVec3]) -> &[[f64; 3]] {
    // SAFETY: the const asserts at module scope prove size + alignment
    // compatibility; DVec3 is #[repr(C)] in glam, so field order xyz matches
    // [f64; 3]. Length is preserved; no aliasing issue (shared borrow).
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<[f64; 3]>(), v.len()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn icosphere_total_area_matches_sphere() {
        // Plan acceptance: subdiv=7 icosphere total area ≈ 4πa² within 0.5%.
        let radius = 10.0;
        let surf = Surface::icosphere(radius, 7);
        let total: f64 = surf.face_areas().iter().sum();
        let expected = 4.0 * PI * radius * radius;
        let rel_err = (total - expected).abs() / expected;
        assert!(rel_err < 0.005, "rel. area error {:e} >= 0.5%", rel_err);
    }

    #[test]
    fn icosphere_face_counts_match_formula() {
        for s in [0usize, 1, 3, 7] {
            let surf = Surface::icosphere(1.0, s);
            assert_eq!(surf.num_faces(), 20 * (s + 1).pow(2));
        }
    }

    #[test]
    fn icosphere_vertices_lie_on_sphere() {
        let r = 5.0;
        let surf = Surface::icosphere(r, 3);
        for v in surf.vertices() {
            let len = DVec3::from(*v).length();
            assert!(
                (len - r).abs() < 1e-12,
                "vertex off sphere: |v|={len} vs r={r}"
            );
        }
    }

    #[test]
    fn icosphere_normals_point_outward() {
        let surf = Surface::icosphere(10.0, 3);
        for (c, n) in surf.face_centroids().iter().zip(surf.face_normals()) {
            let dot = c[0] * n[0] + c[1] * n[1] + c[2] * n[2];
            assert!(dot > 0.0, "inward normal: c·n = {}", dot);
        }
    }

    #[test]
    fn from_mesh_rejects_out_of_range_index() {
        let verts = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = [[0u32, 1, 5]];
        let r = Surface::from_mesh(&verts, &faces);
        assert!(matches!(r, Err(Error::MeshFaceOutOfRange { .. })));
    }

    #[test]
    fn from_mesh_rejects_open_surface() {
        // Four vertices of a tetrahedron but only three faces — last face
        // missing → edges unpaired → not closed.
        let verts = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let faces = [[0u32, 1, 2], [0, 3, 1], [0, 2, 3]];
        let r = Surface::from_mesh(&verts, &faces);
        assert!(matches!(r, Err(Error::NonManifoldMesh { .. })));
    }

    #[test]
    fn classify_charges_inside_sphere() {
        let s = Surface::icosphere(5.0, 3);
        let inside = [[0.0, 0.0, 0.0], [1.0, 2.0, -1.0]];
        assert_eq!(s.classify_charges(&inside).unwrap(), ChargeSide::Interior);
    }

    #[test]
    fn classify_charges_outside_sphere() {
        let s = Surface::icosphere(5.0, 3);
        let outside = [[10.0, 0.0, 0.0], [0.0, 6.0, 0.0]];
        assert_eq!(s.classify_charges(&outside).unwrap(), ChargeSide::Exterior);
    }

    #[test]
    fn classify_charges_rejects_mixed() {
        let s = Surface::icosphere(5.0, 3);
        let mixed = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let r = s.classify_charges(&mixed);
        assert!(matches!(r, Err(Error::MixedChargeSides { .. })));
    }
}
