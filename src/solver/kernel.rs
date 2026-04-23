//! Block-matrix entries for the Juffer BIE, shared between dense
//! assembly (Stage-0 only; no longer built) and the matrix-free operator.
//!
//! Returns `(k0, k0p, kk, kkp)` at collocation pair `(a, b)`:
//! - `k0`, `k0p`: Laplace single- and double-layer panel integrals
//! - `kk`, `kkp`: Yukawa single- and double-layer at the solve's `kappa`.
//!
//! Self-panel `K_κ'` PV vanishes for flat centroid-collocation
//! (`(c − s')·n_b` is identically zero in the panel plane), so the
//! primed entries at `a == b` are returned as `0`.

use crate::geometry::panel::FaceGeoms;
use crate::solver::panel_integrals;

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

    let (k0, k0p) = panel_integrals::k_and_kprime_off(ca, tri, nb, ab, 0.0);
    // why: at κ = 0 the Yukawa and Laplace off-diagonal entries are
    // identical; returning the already-computed Laplace pair avoids
    // a second trip through `k_and_kprime_off` that halves kernel
    // work on salt-free solves.
    if kappa == 0.0 {
        return (k0, k0p, k0, k0p);
    }
    let (kk, kkp) = panel_integrals::k_and_kprime_off(ca, tri, nb, ab, kappa);
    (k0, k0p, kk, kkp)
}
