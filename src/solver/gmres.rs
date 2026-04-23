//! GMRES solve wrapping `faer_gmres` around the matrix-free BEM operator.
//!
//! Memory drops from O(N²) (dense LU) to O(N·k) for a Krylov subspace of
//! dimension k. The Juffer BIE is poorly conditioned without a
//! preconditioner, so we always apply one:
//!
//! - Small meshes (N ≤ [`NEIGHBOR_BLOCK_PANEL_THRESHOLD`]) → cheap 2×2
//!   block-Jacobi. Converges in ~10–20 applies on smooth surfaces.
//! - Large meshes → neighbour-block (RAS) preconditioner that inverts
//!   the local (self + 6 nearest) sub-block per panel. Targets the
//!   heterogeneous-mesh failure mode where block-Jacobi loses diagonal
//!   dominance.
//!
//! `faer_gmres` internally tracks the relative residual `‖r‖₂ / ‖b‖₂`
//! (and applies the preconditioner as left-preconditioning), so
//! `RELATIVE_TOL` is already a relative tolerance.

use crate::error::{Error, Result};
use crate::solver::operator::BemOperator;
use crate::solver::precond::{BlockJacobi, NeighborBlock};
use faer::matrix_free::LinOp;
use faer::{Mat, MatRef};

const RELATIVE_TOL: f64 = 1e-5;
const KRYLOV_DIM: usize = 200;
const MAX_RESTARTS: usize = 5;

// why: below ~3k panels the block-Jacobi preconditioner already
// converges in tens of iterations, and the neighbour-block build
// (O(N²) nearest-neighbour search + N small LUs) costs more than it
// saves. Above this threshold the neighbour-block amortises.
const NEIGHBOR_BLOCK_PANEL_THRESHOLD: usize = 3000;
const NEIGHBOR_BLOCK_K: usize = 6;

pub(super) fn solve(op: BemOperator<'_>, rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    let b = MatRef::from_column_major_slice(&rhs, n, 1);
    let mut x = Mat::<f64>::zeros(n, 1);

    let panels = op.geom.len();
    let jacobi;
    let neighbor;
    let precond: &dyn LinOp<f64> = if panels < NEIGHBOR_BLOCK_PANEL_THRESHOLD {
        jacobi = BlockJacobi::new(op.geom, op.eps_ratio, op.kappa);
        &jacobi
    } else {
        neighbor = NeighborBlock::new(op.geom, op.eps_ratio, op.kappa, NEIGHBOR_BLOCK_K);
        &neighbor
    };

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
    if std::env::var_os("SELFIE_TRACE").is_some() {
        eprintln!("[trace] GMRES converged in {iters} iterations, residual {residual:.3e}");
    }

    let out = x.col_as_slice(0).to_vec();
    if let Some(i) = out.iter().position(|v| !v.is_finite()) {
        return Err(Error::SolveFailed {
            reason: format!("non-finite value at index {i} after GMRES"),
        });
    }
    Ok(out)
}
