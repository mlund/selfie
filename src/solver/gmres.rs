//! GMRES solve wrapping `faer_gmres` around the matrix-free BEM operator.
//!
//! Memory drops from O(N²) (dense LU) to O(N·k) for a Krylov subspace of
//! dimension k. The Juffer BIE is poorly conditioned — unpreconditioned
//! GMRES on lysozyme failed to reach 1e-2 relative residual in 60 iters
//! — so we always apply the analytical block-Jacobi preconditioner
//! `diag(D_a⁻¹)` (see `precond.rs`).
//!
//! `faer_gmres` internally tracks the relative residual `‖r‖₂ / ‖b‖₂`,
//! so the `threshold` argument is already a relative tolerance.

use crate::error::{Error, Result};
use crate::solver::operator::BemOperator;
use crate::solver::precond::BlockJacobi;
use faer::{Mat, MatRef};

// why: 1e-5 matches pygbe and most production PB-BEM codes. Surface
// densities are integrated into energies (O(h²) discretisation error),
// so solver residual below that is wasted work.
// why: 1e-5 is what the existing analytical validation tests need —
// 1e-4 is too loose for the Onsager gap-shrink sensitivity test, as
// measured empirically. Krylov dim 200 matches pygbe's default; it's
// large enough to hold the full preconditioned convergence run
// without a mid-solve restart that truncates accumulated history.
const RELATIVE_TOL: f64 = 1e-5;
const KRYLOV_DIM: usize = 200;
const MAX_RESTARTS: usize = 5;

pub(super) fn solve(op: BemOperator<'_>, rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    let b = MatRef::from_column_major_slice(&rhs, n, 1);
    let mut x = Mat::<f64>::zeros(n, 1);

    // why: the preconditioner needs the same geometry + dielectric
    // parameters as `op`; build it from `op`'s fields so the two can't
    // drift out of sync.
    let precond = BlockJacobi::new(op.geom, op.eps_ratio, op.kappa);

    let (residual, iters) = faer_gmres::restarted_gmres(
        op,
        b,
        x.as_mut(),
        KRYLOV_DIM,
        MAX_RESTARTS,
        RELATIVE_TOL,
        Some(&precond),
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
