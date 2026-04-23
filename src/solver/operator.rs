//! Matrix-free linear operator for the Juffer BIE block system.
//!
//! Implements the action `y = A·x` via the Barnes-Hut treecode in
//! [`crate::solver::treecode`]: far clusters use Taylor-order-2
//! single-layer + order-1 double-layer multipole; near clusters use
//! panel-integrated `kernel::block_entries`. O(N log N) per apply;
//! replaces the O(N²) direct path used up through Stage 3a.
//!
//! At lysozyme scale (N ≈ 14 k) the matrix never materialises — dense
//! LU would demand 6.6 GB.
//!
//! Per-iteration wall-clock spacing is emitted at `log::debug!`.
//! Install an `env_logger` (or similar) and set
//! `RUST_LOG=selfie=debug` to see it.

use crate::geometry::panel::FaceGeoms;
use crate::solver::treecode::PointTreecode;
use faer::dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::{MatMut, MatRef, Par};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

static APPLY_COUNT: AtomicUsize = AtomicUsize::new(0);
static FIRST_APPLY: OnceLock<Instant> = OnceLock::new();

// why: Barnes-Hut MAC parameter. Laplace single- and double-layer
// kernels run on the `P = 6` spherical-harmonic expansion in
// `solver::treecode::solid_harmonic` — error envelope `θ^7`, so
// they're accurate to ~10⁻³ per far cluster even at `θ = 0.6`.
// The Yukawa path still uses the Taylor-order-2 cartesian moments
// in `solver::treecode::multipole`, with error `~θ²`; at `θ = 0.6`
// that's ~0.36, and pushing to `θ = 0.7` sends Kirkwood+salt and
// the exterior reciprocity gate past their 1 % tolerance. Widen
// once Yukawa migrates to the SH basis.
const MAC_THETA: f64 = 0.6;
// Max panels per leaf. Small leaves keep the tree deep enough that
// MAC passes at shallow levels on cluster-in-cluster geometries like
// protein SES. Empirically `n_crit = 50` is the sweet spot between
// direct-sum cost per leaf and tree depth.
const N_CRIT: usize = 50;

#[derive(Debug)]
pub(super) struct BemOperator<'a> {
    pub(super) geom: &'a FaceGeoms,
    pub(super) eps_ratio: f64,
    pub(super) kappa: f64,
    pub(super) tree: PointTreecode<'a>,
}

impl<'a> BemOperator<'a> {
    pub(super) fn new(geom: &'a FaceGeoms, eps_ratio: f64, kappa: f64) -> Self {
        let tree = PointTreecode::new(geom, MAC_THETA, N_CRIT);
        Self {
            geom,
            eps_ratio,
            kappa,
            tree,
        }
    }
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
        let t_apply = log::log_enabled!(log::Level::Debug).then(|| {
            FIRST_APPLY.get_or_init(Instant::now);
            Instant::now()
        });
        let n = self.geom.len();
        let x: Vec<f64> = rhs.col(0).iter().copied().collect();
        let (x_f, x_h) = x.split_at(n);

        let (top, bot) = self
            .tree
            .apply_bem_operator(x_f, x_h, self.kappa, self.eps_ratio);

        for (a, (&t, &b)) in top.iter().zip(&bot).enumerate() {
            out[(a, 0)] = t;
            out[(n + a, 0)] = b;
        }

        if let Some(t) = t_apply {
            let count = APPLY_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
            let since_first = FIRST_APPLY
                .get()
                .map_or(0.0, |t0| t0.elapsed().as_secs_f64());
            log::debug!(
                "apply #{count}: {:.2}s  (cumulative {:.1}s)",
                t.elapsed().as_secs_f64(),
                since_first,
            );
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
