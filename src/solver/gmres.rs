//! GMRES solve wrapping `faer_gmres` around the matrix-free BEM operator.
//!
//! Replaced dense LU in Stage 1. Memory drops from O(N²) to O(N·k) for
//! a Krylov subspace of dimension k, unblocking protein-scale meshes
//! where the dense matrix would be multi-GB.
//!
//! `faer_gmres` internally tracks the relative residual `‖r‖₂ / ‖b‖₂`,
//! so the `threshold` argument is already a relative tolerance — no
//! scaling by `‖b‖` needed here.

use crate::error::{Error, Result};
use crate::solver::operator::BemOperator;
use faer::{Mat, MatRef};

const RELATIVE_TOL: f64 = 1e-8;
const KRYLOV_DIM: usize = 30;
const MAX_RESTARTS: usize = 50;

pub(super) fn solve(op: BemOperator<'_>, rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    let b = MatRef::from_column_major_slice(&rhs, n, 1);
    let mut x = Mat::<f64>::zeros(n, 1);

    let (residual, iters) = faer_gmres::restarted_gmres(
        op,
        b,
        x.as_mut(),
        KRYLOV_DIM,
        MAX_RESTARTS,
        RELATIVE_TOL,
        None,
    )
    .map_err(|e| Error::SolveFailed {
        reason: e.to_string(),
    })?;
    log::info!("GMRES: {iters} iterations, final relative residual {residual:.3e}");

    let out = x.col_as_slice(0).to_vec();
    if let Some(i) = out.iter().position(|v| !v.is_finite()) {
        return Err(Error::SolveFailed {
            reason: format!("non-finite value at index {i} after GMRES"),
        });
    }
    Ok(out)
}
