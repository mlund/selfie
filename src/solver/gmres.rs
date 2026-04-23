//! GMRES solve wrapping `faer_gmres` around the matrix-free BEM
//! operator. Memory drops from O(N²) (dense LU) to O(N·k) for a
//! Krylov subspace of dimension k.
//!
//! The Juffer BIE is poorly conditioned without a preconditioner, so
//! callers always supply one. Preconditioner choice (block-Jacobi vs
//! neighbour-block RAS) lives in [`crate::solver::context`]; this
//! module only consumes the already-built operator and preconditioner
//! so a caller that solves many RHS against the same geometry
//! (e.g. `LinearResponse::precompute`) amortises both builds.
//!
//! `faer_gmres` internally tracks the relative residual `‖r‖₂ / ‖b‖₂`
//! and applies the preconditioner as left-preconditioning, so
//! `RELATIVE_TOL` is already a relative tolerance.

use crate::error::{Error, Result};
use crate::solver::operator::BemOperator;
use faer::matrix_free::LinOp;
use faer::{Mat, MatRef};

const RELATIVE_TOL: f64 = 1e-5;
const KRYLOV_DIM: usize = 200;
const MAX_RESTARTS: usize = 5;

pub(super) fn solve_with(
    op: &BemOperator<'_>,
    precond: &dyn LinOp<f64>,
    rhs: Vec<f64>,
) -> Result<Vec<f64>> {
    let n = rhs.len();
    let b = MatRef::from_column_major_slice(&rhs, n, 1);
    let mut x = Mat::<f64>::zeros(n, 1);

    // why: faer's blanket `impl<M: LinOp<T>> LinOp<T> for &M` lets
    // GMRES take the operator by reference without consuming it, so
    // one context can drive many solves.
    let (residual, iters) = faer_gmres::restarted_gmres(
        op,
        b,
        x.as_mut(),
        KRYLOV_DIM,
        MAX_RESTARTS,
        RELATIVE_TOL,
        Some(precond),
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
