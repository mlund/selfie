use thiserror::Error;

/// All fallible operations in this crate return [`Result<T>`] with this enum.
///
/// Each variant carries structured fields rather than a flat message so a
/// future pyo3 layer can map variants to specific Python exception types
/// (the `MeshInvalid*` / `*Mismatch` family → `ValueError`;
/// `SolveFailed` → `RuntimeError`) and keep the payload introspectable
/// from Python.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error(
        "face {face} references vertex index {index}, but the mesh has only {vertex_count} vertices"
    )]
    MeshFaceOutOfRange {
        face: usize,
        index: u32,
        vertex_count: usize,
    },

    #[error("face {face} is degenerate (area ≈ {area:e})")]
    DegenerateFace { face: usize, area: f64 },

    #[error(
        "face {face} normal is not outward-pointing (centroid·normal = {dot:e}); expected > 0 on a sphere centered at origin"
    )]
    NormalOrientation { face: usize, dot: f64 },

    #[error("charge arrays have mismatched lengths: {positions} positions vs {values} values")]
    ChargeLenMismatch { positions: usize, values: usize },

    #[error("charge index {index} is out of range (have {count} charges)")]
    ChargeIndexOutOfRange { index: usize, count: usize },

    #[error("output buffer too small: expected at least {expected} elements, got {got}")]
    OutputBufferTooSmall { expected: usize, got: usize },

    #[error("linear solve failed: {reason}")]
    SolveFailed { reason: String },

    #[error("I/O reading {path}: {reason}")]
    Io { path: String, reason: String },

    #[error(
        "mesh is not a closed orientable 2-manifold: {reason} \
         (this solver requires a closed surface enclosing a single dielectric volume)"
    )]
    NonManifoldMesh { reason: String },

    #[error(
        "charges straddle the dielectric boundary: {interior} inside, {exterior} outside, {on_surface} on the surface"
    )]
    MixedChargeSides {
        interior: usize,
        exterior: usize,
        on_surface: usize,
    },
}

pub type Result<T> = core::result::Result<T, Error>;
