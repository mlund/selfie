//! Shared geometry and dielectric constants for the sphere-Kirkwood
//! integration tests. Pair scale is salt-bridge-like (≈ 6.7 Å apart), both
//! charges 3 Å above the dielectric boundary.

#![cfg(feature = "validation")]
#![allow(dead_code)] // each test binary only uses a subset.

pub const A: f64 = 10.0; // sphere radius (Å)
pub const R: f64 = 13.0; // charge/eval radius (Å) — 3 Å standoff.
pub const EPS_IN: f64 = 2.0;
pub const EPS_OUT: f64 = 80.0;

/// Source point at `[R, 0, 0]`.
pub fn source() -> [f64; 3] {
    [R, 0.0, 0.0]
}

/// Evaluation point at γ = 30° from the source on the same |r| = R circle.
/// Chord length `2 R sin(15°) ≈ 6.73 Å`.
pub fn eval_point() -> [f64; 3] {
    [R * (3f64.sqrt() / 2.0), R * 0.5, 0.0]
}
