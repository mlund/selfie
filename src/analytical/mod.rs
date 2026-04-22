//! Analytical references — test oracles, not product surface.
//!
//! Only compiled under `#[cfg(test)]` or the `validation` feature.

// Numerical utilities only used by the Kirkwood analytical references —
// not part of the product surface, so kept crate-private even under the
// `validation` feature.
pub(crate) mod bessel;
mod legendre;

pub mod kirkwood;
pub mod kirkwood_inside;
pub mod kirkwood_salt;
