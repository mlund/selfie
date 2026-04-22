//! Boundary Element Method solver for the linearized Poisson–Boltzmann equation.
//!
//! See `bem_pb_plan.md` in the repo root for the math, sign conventions, and
//! phase plan. The user-facing vocabulary is intentionally minimal:
//! [`Surface`], [`Dielectric`], [`BemSolution`], and [`Error`].

mod error;
mod geometry;
mod solver;

// why: units exposes conversion helpers (kcal/mol, kT) that callers reach
// through `selfie::units::to_kcal_per_mol` etc. Keeping the core type
// [`Dielectric`] re-exported at the crate root for convenience while leaving
// conversions namespaced preserves the "few public nouns" principle.
pub mod units;

// Analytical references (Kirkwood) are a test oracle — not product surface.
// Public only under the `validation` feature or in `cfg(test)`, so integration
// tests can import it without polluting the default public API.
#[cfg(any(test, feature = "validation"))]
pub mod analytical;

pub use error::{Error, Result};
pub use geometry::Surface;
pub use solver::BemSolution;
pub use units::Dielectric;
