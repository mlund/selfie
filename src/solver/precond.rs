//! Block-Jacobi preconditioner for the Juffer BIE.
//!
//! The panel-diagonal 2×2 block at row-pair `a` is
//!
//! ```text
//! D_a = [ 0.5            −ε_ratio · K₀_self(a) ]
//!       [ 0.5             K_κ_self(a)          ]
//! ```
//!
//! (the primed `K'` self-panel PVs vanish for flat centroid-collocation —
//! see `panel_integrals::k_self` / `wrg_g0_self`). `D_a` is analytically
//! invertible; applying `diag(D_a⁻¹)` costs O(N) per call and typically
//! contracts the Juffer operator's spectrum into a tight cluster near 1,
//! dropping GMRES iteration count by several-fold. Unpreconditioned
//! GMRES on lysozyme (14.4 k faces, κ = 0.125 Å⁻¹) failed to reach
//! 1e-2 relative residual in 60 iterations; pygbe's published
//! iteration count with this same preconditioner is ≈ 33–39.

use crate::geometry::panel::FaceGeoms;
use crate::solver::kernel::block_entries;
use faer::dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::{MatMut, MatRef, Par};
use rayon::prelude::*;

#[derive(Debug)]
pub(super) struct BlockJacobi {
    // Stored row-major: [d00, d01, d10, d11] per panel.
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
                [
                    inv_det * a22,
                    -inv_det * a12,
                    -inv_det * a21,
                    inv_det * a11,
                ]
            })
            .collect();
        Self { inv_diag }
    }

    fn n(&self) -> usize {
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
        // O(N) no-heap: scan the 2N vector panel by panel, applying
        // the cached 2×2 inverse in place.
        for (a, &[d00, d01, d10, d11]) in self.inv_diag.iter().enumerate() {
            let r_top = rhs[(a, 0)];
            let r_bot = rhs[(n + a, 0)];
            out[(a, 0)] = d00.mul_add(r_top, d01 * r_bot);
            out[(n + a, 0)] = d10.mul_add(r_top, d11 * r_bot);
        }
    }

    fn conj_apply(&self, _: MatMut<f64>, _: MatRef<f64>, _: Par, _: &mut MemStack) {
        unreachable!("conj_apply is not called by GMRES on the Juffer preconditioner");
    }
}
