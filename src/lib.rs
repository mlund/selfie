//! Boundary Element Method solver for the linearized Poisson–Boltzmann equation.
//!
//! Juffer derivative-BIE formulation with centroid collocation and 3-point
//! barycentric Gauss quadrature. User-facing vocabulary is intentionally
//! minimal: [`Surface`], [`Dielectric`], [`BemSolution`], and [`Error`].

mod error;
mod geometry;
pub mod io;
mod linear_response;
#[cfg(feature = "python")]
mod python;
mod solver;

// why: units exposes conversion helpers (kJ/mol, kT) that callers reach
// through `selfie::units::to_kJ_per_mol` etc. Keeping the core type
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
pub use linear_response::LinearResponse;
pub use solver::BemSolution;
pub use units::{ChargeSide, Dielectric};
