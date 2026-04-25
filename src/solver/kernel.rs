//! Block-matrix entries for the Juffer BIE, shared by the matrix-free
//! operator and its block-Jacobi preconditioner.
//!
//! Returns `(k0, k0p, kk, kkp)` at collocation pair `(a, b)`:
//! - `k0`, `k0p`: Laplace single- and double-layer panel integrals
//! - `kk`, `kkp`: Yukawa single- and double-layer at the solve's `kappa`.
//!
//! Self-panel `K_κ'` PV vanishes for flat centroid-collocation
//! (`(c − s')·n_b` is identically zero in the panel plane), so the
//! primed entries at `a == b` are returned as `0`.
//!
//! Off-diagonal entries use adaptive quadrature: 3-point Gauss when the
//! observer is far from the source panel (smooth kernel), 7-point
//! Dunavant when the observer sits within a few panel-widths (quasi-
//! singular kernel — 3-point under-integrates and degrades GMRES
//! conditioning on heterogeneous protein meshes).

use crate::geometry::panel::FaceGeoms;
use crate::solver::panel_integrals;

// why: "near" = observer centroid is closer than this multiple of
// √area_b to the source centroid. At 2.5 the 3-point rule's relative
// error on 1/r is ~1e-3 per near-neighbour entry; pushing the
// threshold higher eats into the 7-point cost budget without
// measurably helping condition number. The constant is a geometric
// ratio, not dimensional — unitless by construction.
const NEAR_RATIO: f64 = 2.5;

pub(super) fn block_entries(
    geom: &FaceGeoms,
    a: usize,
    b: usize,
    kappa: f64,
) -> (f64, f64, f64, f64) {
    let (k0, k0p) = block_entries_laplace(geom, a, b);
    // why: at κ = 0 the Yukawa and Laplace off-diagonal entries are
    // identical; returning the already-computed Laplace pair avoids
    // a second quadrature that halves kernel work on salt-free solves.
    if kappa == 0.0 {
        return (k0, k0p, k0, k0p);
    }
    let (kk, kkp) = block_entries_yukawa(geom, a, b, kappa);
    (k0, k0p, kk, kkp)
}

/// Laplace-only `(K_0, K'_0)` block entries. Matches the first two
/// components of [`block_entries`] but skips any Yukawa work —
/// suitable for the Laplace traversal of the asymmetric-MAC
/// treecode, which runs at a looser θ than the Yukawa walk.
pub(super) fn block_entries_laplace(geom: &FaceGeoms, a: usize, b: usize) -> (f64, f64) {
    let ca = geom.centroids[a];
    if a == b {
        return (panel_integrals::k_self(geom.tris[b], ca, 0.0), 0.0);
    }
    eval_off_pair(geom, ca, b, 0.0)
}

/// Yukawa-only `(K_κ, K'_κ)` block entries. Debug-asserts `κ > 0`;
/// for `κ = 0` the result would duplicate [`block_entries_laplace`],
/// and the treecode only invokes this path when screening is active.
pub(super) fn block_entries_yukawa(geom: &FaceGeoms, a: usize, b: usize, kappa: f64) -> (f64, f64) {
    debug_assert!(kappa > 0.0);
    let ca = geom.centroids[a];
    if a == b {
        return (panel_integrals::k_self(geom.tris[b], ca, kappa), 0.0);
    }
    eval_off_pair(geom, ca, b, kappa)
}

#[inline]
fn eval_off_pair(geom: &FaceGeoms, ca: glam::DVec3, b: usize, kappa: f64) -> (f64, f64) {
    let nb = geom.normals[b];
    let ab = geom.areas[b];
    let tri = geom.tris[b];
    let dist_sq = (ca - geom.centroids[b]).length_squared();
    if dist_sq < NEAR_RATIO * NEAR_RATIO * ab {
        panel_integrals::k_and_kprime_near(ca, tri, nb, ab, kappa)
    } else {
        panel_integrals::k_and_kprime_off(ca, tri, nb, ab, kappa)
    }
}
