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
    let ca = geom.centroids[a];
    let nb = geom.normals[b];
    let ab = geom.areas[b];
    let tri = geom.tris[b];

    if a == b {
        let k0 = panel_integrals::k_self(tri, ca, 0.0);
        // why: in a salt-free solve the Yukawa self-panel integral is
        // exactly the Laplace one — skip the duplicate WRG evaluation.
        let kk = if kappa == 0.0 {
            k0
        } else {
            panel_integrals::k_self(tri, ca, kappa)
        };
        return (k0, 0.0, kk, 0.0);
    }

    let dist_sq = (ca - geom.centroids[b]).length_squared();
    let is_near = dist_sq < NEAR_RATIO * NEAR_RATIO * ab;
    let eval_off = |kappa| {
        if is_near {
            panel_integrals::k_and_kprime_near(ca, tri, nb, ab, kappa)
        } else {
            panel_integrals::k_and_kprime_off(ca, tri, nb, ab, kappa)
        }
    };

    let (k0, k0p) = eval_off(0.0);
    // why: at κ = 0 the Yukawa and Laplace off-diagonal entries are
    // identical; returning the already-computed Laplace pair avoids
    // a second quadrature that halves kernel work on salt-free solves.
    if kappa == 0.0 {
        return (k0, k0p, k0, k0p);
    }
    let (kk, kkp) = eval_off(kappa);
    (k0, k0p, kk, kkp)
}
