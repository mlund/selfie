//! Adapter around `hexasphere` 18.0.
//!
//! hexasphere returns f32 `Vec3A` points on the unit sphere with CCW-outward
//! triangles. We convert to f64 once, then renormalize to the exact target
//! radius in f64 — this both erases f32 rounding and, importantly for BEM,
//! puts every vertex on the true sphere so the mesh is the actual surface we
//! ran our Kirkwood analytic solution on (not an inscribed polyhedron). The
//! plan calls this out as the fix for the "faceting bias" failure mode.

use glam::DVec3;
use hexasphere::shapes::IcoSphere;

pub fn build(radius: f64, subdivisions: usize) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    let sphere = IcoSphere::new(subdivisions, |_| ());

    let vertices: Vec<DVec3> = sphere
        .raw_points()
        .iter()
        .map(|p| {
            // why: hexasphere emits unit-sphere points. We *renormalize after
            // f32→f64 conversion* rather than just multiplying by radius, so
            // any f32 drift from slerp subdivision is projected back onto the
            // sphere at f64 precision.
            p.as_dvec3().normalize() * radius
        })
        .collect();

    let flat = sphere.get_all_indices();
    debug_assert_eq!(
        flat.len() % 3,
        0,
        "hexasphere index buffer not divisible by 3"
    );
    let faces: Vec<[u32; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();

    (vertices, faces)
}
