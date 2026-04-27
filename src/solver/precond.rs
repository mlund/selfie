//! Preconditioners for the Juffer BIE GMRES solve.
//!
//! Two variants:
//!
//! - [`BlockJacobi`] — per-panel 2×2 diagonal inverse. Cheap O(N)
//!   build + apply. Adequate for spheres / smooth convex surfaces.
//! - [`NeighborBlock`] — per-panel 2(k+1)×2(k+1) inverse of the
//!   panel and its `k` nearest neighbours. Restricted additive Schwarz
//!   (RAS): each panel scatters only its own self-row back into the
//!   output, so the apply is parallel-friendly without atomics.
//!   Attacks the failure mode of block-Jacobi on heterogeneous
//!   protein meshes, where the 2×2 diagonal loses dominance against
//!   the many close-neighbour off-diagonal blocks.

use crate::geometry::panel::FaceGeoms;
use crate::solver::kernel::block_entries;
use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::solvers::{PartialPivLu, Solve};
use faer::matrix_free::LinOp;
use faer::{Mat, MatMut, MatRef, Par};
use rayon::prelude::*;
use std::fmt;

#[derive(Debug)]
pub(super) struct BlockJacobi {
    // [d00, d01, d10, d11] per panel — the inverse 2×2 block.
    inv_diag: Vec<[f64; 4]>,
}

impl BlockJacobi {
    pub(super) fn new(geom: &FaceGeoms, eps_ratio: f64, kappa: f64) -> Self {
        let inv_diag = (0..geom.len())
            .into_par_iter()
            .map(|a| {
                let (k0_self, _, kk_self, _) = block_entries(geom, a, a, kappa);
                // D_a = [[0.5, -ε·k0], [0.5, kk]], det = 0.5·(kk + ε·k0).
                // Both kk_self and k0_self are positive single-layer
                // integrals, so det > 0 for any physical ε_ratio > 0.
                let a11 = 0.5_f64;
                let a12 = -eps_ratio * k0_self;
                let a21 = 0.5_f64;
                let a22 = kk_self;
                let inv_det = 1.0 / a11.mul_add(a22, -a12 * a21);
                [inv_det * a22, -inv_det * a12, -inv_det * a21, inv_det * a11]
            })
            .collect();
        Self { inv_diag }
    }

    const fn n(&self) -> usize {
        self.inv_diag.len()
    }
}

impl LinOp<f64> for BlockJacobi {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::empty()
    }
    fn nrows(&self) -> usize {
        2 * self.n()
    }
    fn ncols(&self) -> usize {
        2 * self.n()
    }
    fn apply(&self, mut out: MatMut<f64>, rhs: MatRef<f64>, _par: Par, _stack: &mut MemStack) {
        let n = self.n();
        for (a, &[d00, d01, d10, d11]) in self.inv_diag.iter().enumerate() {
            let r_top = rhs[(a, 0)];
            let r_bot = rhs[(n + a, 0)];
            out[(a, 0)] = d00.mul_add(r_top, d01 * r_bot);
            out[(n + a, 0)] = d10.mul_add(r_top, d11 * r_bot);
        }
    }
    fn conj_apply(&self, out: MatMut<f64>, rhs: MatRef<f64>, par: Par, stack: &mut MemStack) {
        // why: real-valued LinOp — conjugate equals the operator
        // itself, so delegating is mathematically correct.
        self.apply(out, rhs, par, stack);
    }
}

/// Restricted Additive Schwarz (RAS) preconditioner.
///
/// For each panel `a` we pick `k` nearest panels (by centroid distance)
/// and LU-factor the full 2(k+1)×2(k+1) local sub-block of the Juffer
/// operator restricted to `{a, nbr_1, …, nbr_k}`. At apply time the
/// local system is solved; only the two self-panel components of the
/// solution are written back to the global output vector. The "R" in
/// RAS (vs. standard additive Schwarz) is precisely this restricted
/// scatter — it avoids overlapping writes so rayon can parallelise the
/// outer loop without atomics, and in practice matches or beats
/// unrestricted ASM (Cai–Sarkis 1999).
pub(super) struct NeighborBlock {
    k: usize,
    // Flat (n × (k+1)) indices: neighbourhoods[a·(k+1) .. (a+1)·(k+1)].
    // Position 0 is the panel itself; 1..=k are its k nearest neighbours.
    neighbourhoods: Vec<u32>,
    // LU factor of the 2(k+1)×2(k+1) local block per panel.
    local_lu: Vec<PartialPivLu<f64>>,
}

impl fmt::Debug for NeighborBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeighborBlock")
            .field("n", &self.local_lu.len())
            .field("k", &self.k)
            .finish()
    }
}

impl NeighborBlock {
    pub(super) fn new(geom: &FaceGeoms, eps_ratio: f64, kappa: f64, k: usize) -> Self {
        let n = geom.len();
        debug_assert!(k < n, "k neighbours must be < N panels");

        // why: O(N log N) kd-tree build + O(log N) per query beats the
        // hand-rolled O(N²) partial-sort that used to live here; at
        // N=14 k the saving is ~1 s of preconditioner-build time.
        let mut tree: kdtree::KdTree<f64, u32, [f64; 3]> = kdtree::KdTree::new(3);
        for (i, c) in geom.centroids.iter().enumerate() {
            let result = tree.add(c.to_array(), i as u32);
            debug_assert!(result.is_ok(), "kdtree::add failed: {result:?}");
            // SAFETY: `KdTree::add` only fails on `WrongDimension`,
            // `ZeroCapacity`, or `NonFiniteCoordinate`. The first two
            // are statically impossible (tree built with `new(3)` and
            // we feed `[f64; 3]` keys); centroids are finite by mesh
            // construction. The `debug_assert!` above surfaces any
            // future regression in test builds.
            unsafe { result.unwrap_unchecked() };
        }

        // `nearest(p, k+1, …)` returns the query point itself first
        // (distance 0); skip it so only true neighbours enter the
        // local sub-block.
        let neighbourhoods: Vec<u32> = (0..n)
            .into_par_iter()
            .flat_map_iter(|a| {
                let ca = geom.centroids[a].to_array();
                let hits = tree.nearest(&ca, k + 1, &kdtree::distance::squared_euclidean);
                debug_assert!(hits.is_ok(), "kdtree::nearest failed: {hits:?}");
                // SAFETY: `nearest` errors only on `WrongDimension`
                // (impossible — 3-dim tree, 3-dim query) or `Empty`
                // (excluded by the `debug_assert!(k < n)` at entry,
                // so the tree has ≥ k+1 entries).
                let hits = unsafe { hits.unwrap_unchecked() };
                std::iter::once(a as u32).chain(
                    hits.into_iter()
                        .map(|(_, &idx)| idx)
                        .filter(move |&idx| idx != a as u32)
                        .take(k),
                )
            })
            .collect();

        let stride = k + 1;
        let local_lu: Vec<PartialPivLu<f64>> = (0..n)
            .into_par_iter()
            .map(|a| {
                let nbrs = &neighbourhoods[a * stride..(a + 1) * stride];
                let local_size = 2 * stride;
                // Layout: row/col 2i = top of S_a[i], 2i+1 = bot of S_a[i].
                let mat = Mat::<f64>::from_fn(local_size, local_size, |i, j| {
                    let pa = nbrs[i / 2] as usize;
                    let pb = nbrs[j / 2] as usize;
                    let (k0, k0p, kk, kkp) = block_entries(geom, pa, pb, kappa);
                    let half = if pa == pb { 0.5 } else { 0.0 };
                    // why: matching on `(bool, bool)` is exhaustive at
                    // four arms, so no `_ => unreachable!()` panic
                    // branch survives in release.
                    match (i % 2 == 0, j % 2 == 0) {
                        (true, true) => half + k0p,
                        (true, false) => -eps_ratio * k0,
                        (false, true) => half - kkp,
                        (false, false) => kk,
                    }
                });
                mat.partial_piv_lu()
            })
            .collect();

        Self {
            k,
            neighbourhoods,
            local_lu,
        }
    }

    const fn n(&self) -> usize {
        self.local_lu.len()
    }
}

impl LinOp<f64> for NeighborBlock {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::empty()
    }
    fn nrows(&self) -> usize {
        2 * self.n()
    }
    fn ncols(&self) -> usize {
        2 * self.n()
    }
    fn apply(&self, mut out: MatMut<f64>, rhs: MatRef<f64>, _par: Par, _stack: &mut MemStack) {
        let n = self.n();
        let stride = self.k + 1;
        let local_size = 2 * stride;
        // why: the self-rows from every local solve are independent
        // (RAS), so we parallelise over panels; each (top, bot) pair
        // lands in its own two global slots with no contention.
        let solved: Vec<(f64, f64)> = (0..n)
            .into_par_iter()
            .map(|a| {
                let nbrs = &self.neighbourhoods[a * stride..(a + 1) * stride];
                let mut r_local = Mat::<f64>::zeros(local_size, 1);
                for (i, &p) in nbrs.iter().enumerate() {
                    r_local[(2 * i, 0)] = rhs[(p as usize, 0)];
                    r_local[(2 * i + 1, 0)] = rhs[(n + p as usize, 0)];
                }
                self.local_lu[a].solve_in_place(&mut r_local);
                // Position 0 in nbrs is `a` itself; pick its two entries.
                (r_local[(0, 0)], r_local[(1, 0)])
            })
            .collect();

        for (a, &(top, bot)) in solved.iter().enumerate() {
            out[(a, 0)] = top;
            out[(n + a, 0)] = bot;
        }
    }
    fn conj_apply(&self, out: MatMut<f64>, rhs: MatRef<f64>, par: Par, stack: &mut MemStack) {
        // why: real-valued LinOp — conjugate equals the operator
        // itself, so delegating is mathematically correct.
        self.apply(out, rhs, par, stack);
    }
}
