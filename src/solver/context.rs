//! Reusable solver context — a `BemOperator` and preconditioner built
//! once from geometry + dielectric, applied against any number of
//! charge-dependent RHS vectors.
//!
//! `BemSolution::solve` is a one-shot convenience wrapper over this.
//! `LinearResponse::precompute` is the motivating caller: on the
//! lysozyme mesh the RAS preconditioner is a sizeable fraction of
//! per-solve cost and the mesh is identical across all N basis
//! solves, so amortising the build pays linearly in N.
//!
//! Crate-internal. The only external-facing path is still
//! [`BemSolution::solve`](crate::BemSolution::solve).
//!
//! Preconditioner choice:
//!
//! - Small meshes (N ≤ `NEIGHBOR_BLOCK_PANEL_THRESHOLD`) use cheap
//!   2×2 block-Jacobi.
//! - Large meshes use the neighbour-block RAS preconditioner.
//!
//! The threshold matches the one historically chosen inside
//! `gmres::solve`; splitting the build out of that function is the
//! only change.

use crate::error::{Error, Result};
use crate::geometry::Surface;
use crate::solver::assembly;
use crate::solver::gmres;
use crate::solver::operator::BemOperator;
use crate::solver::precond::{BlockJacobi, NeighborBlock};
use crate::units::{ChargeSide, Dielectric};
use faer::matrix_free::LinOp;

const NEIGHBOR_BLOCK_PANEL_THRESHOLD: usize = 3000;
const NEIGHBOR_BLOCK_K: usize = 6;

pub(crate) struct SolveContext<'s> {
    surface: &'s Surface,
    media: Dielectric,
    side: ChargeSide,
    operator: BemOperator<'s>,
    precond: Preconditioner,
}

enum Preconditioner {
    BlockJacobi(BlockJacobi),
    NeighborBlock(NeighborBlock),
}

impl Preconditioner {
    fn as_linop(&self) -> &dyn LinOp<f64> {
        match self {
            Self::BlockJacobi(p) => p,
            Self::NeighborBlock(p) => p,
        }
    }
}

impl<'s> SolveContext<'s> {
    /// Build the operator + preconditioner for this geometry. Panel
    /// count chooses block-Jacobi vs neighbour-block RAS.
    pub(crate) fn new(surface: &'s Surface, media: Dielectric, side: ChargeSide) -> Self {
        let geom = surface.geom_internal();
        let eps_ratio = media.eps_out / media.eps_in;
        let operator = BemOperator::new(geom, eps_ratio, media.kappa);
        let precond = if geom.len() < NEIGHBOR_BLOCK_PANEL_THRESHOLD {
            Preconditioner::BlockJacobi(BlockJacobi::new(geom, eps_ratio, media.kappa))
        } else {
            Preconditioner::NeighborBlock(NeighborBlock::new(
                geom,
                eps_ratio,
                media.kappa,
                NEIGHBOR_BLOCK_K,
            ))
        };
        Self {
            surface,
            media,
            side,
            operator,
            precond,
        }
    }

    /// Solve for `(f, h)` given a charge configuration. The operator
    /// and preconditioner are reused; only the RHS is rebuilt.
    pub(crate) fn solve_charges(
        &self,
        charge_positions: &[[f64; 3]],
        charge_values: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        if charge_positions.len() != charge_values.len() {
            return Err(Error::ChargeLenMismatch {
                positions: charge_positions.len(),
                values: charge_values.len(),
            });
        }
        let rhs = assembly::build_rhs(
            self.surface,
            self.media,
            self.side,
            charge_positions,
            charge_values,
        );
        let mut f = gmres::solve_with(&self.operator, self.precond.as_linop(), rhs)?;
        // why: block ordering in assembly is `[f; h]`; split_off
        // moves ownership of the second half out in O(1) without a
        // heap copy, matching the pattern in `BemSolution::solve`.
        let h = f.split_off(self.surface.num_faces());
        Ok((f, h))
    }
}
