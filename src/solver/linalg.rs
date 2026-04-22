//! Thin wrapper around faer's dense LU.
//!
//! Kept as its own module so a future switch to GMRES (matrix-free, Phase 8)
//! or a different backend doesn't ripple through `assembly.rs` or the public
//! [`crate::BemSolution`] API.

use crate::error::{Error, Result};
use faer::Mat;
use faer::linalg::solvers::Solve;

/// Solve `A x = b` for a square dense real system. Returns `x` as a `Vec<f64>`.
///
/// Uses partial-pivot LU. Phase 1 meshes (≤ 1280 triangles → A is 2560×2560)
/// factor in under a second; see plan doc for the upper mesh size this
/// scales to. Beyond that, switch to matrix-free GMRES (Phase 8).
pub fn solve_dense(a: Mat<f64>, b: Vec<f64>) -> Result<Vec<f64>> {
    let n = a.nrows();
    debug_assert_eq!(a.nrows(), a.ncols());
    debug_assert_eq!(b.len(), n);

    let mut rhs = Mat::<f64>::from_fn(n, 1, |i, _| b[i]);

    let lu = a.partial_piv_lu();
    // why: partial_piv_lu always succeeds structurally (even for singular A,
    // you get a zero pivot). We don't currently detect the singular case;
    // the plan's condition-number sanity check at assembly time is our
    // guardrail. If singular A becomes a real failure mode, check min pivot
    // here and return SolveFailed.
    lu.solve_in_place(&mut rhs);

    // Extract column 0 into a Vec<f64>.
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = rhs[(i, 0)];
        if !v.is_finite() {
            return Err(Error::SolveFailed {
                reason: format!("non-finite value at index {i} after LU solve"),
            });
        }
        out.push(v);
    }
    Ok(out)
}
