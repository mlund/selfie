//! Matrix-free linear operator for the Juffer BIE block system.
//!
//! Implements the action `y = A·x` without ever materialising `A`, which
//! at lysozyme scale (N ≈ 14 k faces → 2N × 2N dense system ≈ 6.6 GB)
//! is the difference between "fits on a laptop" and "doesn't". Each
//! `apply` costs O(N²) kernel evaluations — the same work as one row
//! of the old dense assembly, now spread across rayon-parallel rows.

use crate::geometry::panel::FaceGeoms;
use crate::solver::kernel::block_entries;
use faer::dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::{MatMut, MatRef, Par};
use rayon::prelude::*;

#[derive(Debug)]
pub(super) struct BemOperator<'a> {
    pub(super) geom: &'a FaceGeoms,
    pub(super) eps_ratio: f64,
    pub(super) kappa: f64,
}

impl LinOp<f64> for BemOperator<'_> {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::empty()
    }

    fn nrows(&self) -> usize {
        2 * self.geom.len()
    }

    fn ncols(&self) -> usize {
        2 * self.geom.len()
    }

    fn apply(&self, mut out: MatMut<f64>, rhs: MatRef<f64>, _par: Par, _stack: &mut MemStack) {
        let n = self.geom.len();
        // why: materialise the column as a flat `&[f64]` once so the
        // parallel closure does slice indexing (no bounds-check per
        // entry against MatRef row/col shape).
        let x: Vec<f64> = rhs.col(0).iter().copied().collect();
        let (x_f, x_h) = x.split_at(n);

        // y[a]     = Σ_b (½δ_ab + k0p)·f_b + (−ε_ratio·k0)·h_b
        // y[n+a]   = Σ_b (½δ_ab − kkp)·f_b + kk·h_b
        let y: Vec<(f64, f64)> = (0..n)
            .into_par_iter()
            .map(|a| {
                // Self-panel entry once, outside the b-loop: k'-primed
                // blocks vanish for flat centroid-collocation, so the
                // self contribution is ½I ± 0 on the diagonal plus the
                // single- and double-layer K terms from block_entries.
                let (k0_aa, _, kk_aa, _) = block_entries(self.geom, a, a, self.kappa);
                let mut top = 0.5f64.mul_add(x_f[a], -self.eps_ratio * k0_aa * x_h[a]);
                let mut bot = 0.5f64.mul_add(x_f[a], kk_aa * x_h[a]);
                for b in 0..n {
                    if b == a {
                        continue;
                    }
                    let (k0, k0p, kk, kkp) = block_entries(self.geom, a, b, self.kappa);
                    top += k0p.mul_add(x_f[b], -self.eps_ratio * k0 * x_h[b]);
                    bot += (-kkp).mul_add(x_f[b], kk * x_h[b]);
                }
                (top, bot)
            })
            .collect();

        for (a, &(top, bot)) in y.iter().enumerate() {
            out[(a, 0)] = top;
            out[(n + a, 0)] = bot;
        }
    }

    fn conj_apply(&self, _: MatMut<f64>, _: MatRef<f64>, _: Par, _: &mut MemStack) {
        // faer_gmres's Arnoldi iteration only ever calls `apply`, so
        // this path is unreachable on the non-self-adjoint Juffer
        // operator. If a BiCGStab / normal-equations solver is ever
        // plugged in, implement A^T · x here.
        unreachable!("conj_apply is not called by GMRES on the Juffer operator");
    }
}
