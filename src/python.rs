//! pyo3 bindings for the BEMtzmann BEM solver.
//!
//! Exposes a deliberately minimal Python surface: four classes
//! (`Surface`, `Dielectric`, `BemSolution`, `LinearResponse`), one
//! enum (`ChargeSide`), and a few free functions (`read_pqr`,
//! `read_xyz`, `to_kJ_per_mol`, `to_kt`). Geometry accessors,
//! raw-density getters, and dielectric passthroughs stay on the
//! Rust side — Python workflows rarely need them.
//!
//! Lifetime note: `BemSolution<'s>` and `LinearResponse<'s>` both
//! borrow `&Surface`, which `#[pyclass]` cannot carry. The Python
//! wrappers own `Arc<Surface>` so the backing mesh lives as long
//! as any dependent solution; each method reconstructs a transient
//! borrow-based view from the Arc when it needs to dispatch into
//! the Rust API. This keeps the Rust public surface unchanged:
//! the one crate-internal hook we rely on is
//! `BemSolution::from_densities` (`pub(crate)`).

use crate::error::Error;
use crate::solver::BemSolution;
use crate::units::{ChargeSide, Dielectric};
use crate::{LinearResponse, Surface};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

/// Map `bemtzmann::Error` to the Python exception that best fits each
/// caller-visible misuse.
impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        let msg = err.to_string();
        match err {
            Error::ChargeIndexOutOfRange { .. } => PyIndexError::new_err(msg),
            Error::SolveFailed { .. } => PyRuntimeError::new_err(msg),
            Error::Io { .. } => PyOSError::new_err(msg),
            _ => PyValueError::new_err(msg),
        }
    }
}

fn positions_from_numpy(arr: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "expected a (N, 3) float64 array, got shape {shape:?}"
        )));
    }
    let view = arr.as_array();
    Ok(view
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect())
}

fn faces_from_numpy(arr: &PyReadonlyArray2<'_, u32>) -> PyResult<Vec<[u32; 3]>> {
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "expected a (N, 3) uint32 array, got shape {shape:?}"
        )));
    }
    let view = arr.as_array();
    Ok(view
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect())
}

fn positions_to_numpy<'py>(py: Python<'py>, positions: &[[f64; 3]]) -> Bound<'py, PyArray2<f64>> {
    let n = positions.len();
    let flat: Vec<f64> = positions.iter().flat_map(|p| p.iter().copied()).collect();
    PyArray1::from_vec(py, flat)
        .reshape([n, 3])
        .expect("shape is (N, 3)")
}

// =========================================================================
// ChargeSide
// =========================================================================

#[pyclass(name = "ChargeSide", module = "bemtzmann", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyChargeSide {
    Interior,
    Exterior,
}

impl From<PyChargeSide> for ChargeSide {
    fn from(p: PyChargeSide) -> Self {
        match p {
            PyChargeSide::Interior => ChargeSide::Interior,
            PyChargeSide::Exterior => ChargeSide::Exterior,
        }
    }
}

impl From<ChargeSide> for PyChargeSide {
    fn from(c: ChargeSide) -> Self {
        match c {
            ChargeSide::Interior => PyChargeSide::Interior,
            ChargeSide::Exterior => PyChargeSide::Exterior,
        }
    }
}

// =========================================================================
// Dielectric
// =========================================================================

#[pyclass(name = "Dielectric", module = "bemtzmann", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct PyDielectric {
    inner: Dielectric,
}

#[pymethods]
impl PyDielectric {
    #[new]
    #[pyo3(signature = (eps_in, eps_out, kappa = 0.0))]
    fn new(eps_in: f64, eps_out: f64, kappa: f64) -> Self {
        Self {
            inner: Dielectric::continuum_with_salt(eps_in, eps_out, kappa),
        }
    }

    #[getter]
    fn eps_in(&self) -> f64 {
        self.inner.eps_in
    }

    #[getter]
    fn eps_out(&self) -> f64 {
        self.inner.eps_out
    }

    #[getter]
    fn kappa(&self) -> f64 {
        self.inner.kappa
    }

    fn __repr__(&self) -> String {
        format!(
            "Dielectric(eps_in={}, eps_out={}, kappa={})",
            self.inner.eps_in, self.inner.eps_out, self.inner.kappa
        )
    }
}

// =========================================================================
// Surface
// =========================================================================

#[pyclass(name = "Surface", module = "bemtzmann", frozen)]
pub struct PySurface {
    inner: Arc<Surface>,
}

#[pymethods]
impl PySurface {
    /// Regular icosphere mesh of given radius and subdivision depth.
    #[staticmethod]
    fn icosphere(radius: f64, subdivisions: usize) -> Self {
        Self {
            inner: Arc::new(Surface::icosphere(radius, subdivisions)),
        }
    }

    /// Build a surface from `(vertices, faces)` numpy arrays.
    ///
    /// `vertices` is `(V, 3) float64`; `faces` is `(F, 3) uint32`
    /// and must reference a closed, orientable, outward-normalled 2-manifold.
    #[staticmethod]
    fn from_mesh(
        vertices: PyReadonlyArray2<'_, f64>,
        faces: PyReadonlyArray2<'_, u32>,
    ) -> PyResult<Self> {
        let verts = positions_from_numpy(&vertices)?;
        let tris = faces_from_numpy(&faces)?;
        let surface = Surface::from_mesh(&verts, &tris)?;
        Ok(Self {
            inner: Arc::new(surface),
        })
    }

    /// Convenience: load an MSMS `.vert` + `.face` pair into a
    /// Surface in one call.
    #[staticmethod]
    fn from_msms(vert_path: &str, face_path: &str) -> PyResult<Self> {
        let (vertices, faces) = crate::io::read_msms(vert_path, face_path)?;
        let surface = Surface::from_mesh(&vertices, &faces)?;
        Ok(Self {
            inner: Arc::new(surface),
        })
    }

    /// Build a closed Gaussian molecular surface from atom positions and
    /// radii using the in-tree marching-cubes mesher (no NanoShaper or
    /// MSMS dependency).
    ///
    /// `positions` is `(N, 3) float64`, `radii` is `(N,) float64` with all
    /// values positive and finite. `grid_spacing` is in Å (typical 0.3–1.0;
    /// smaller is finer with quadratic memory cost).
    #[cfg(feature = "mesh")]
    #[staticmethod]
    fn from_atoms_gaussian(
        positions: PyReadonlyArray2<'_, f64>,
        radii: PyReadonlyArray1<'_, f64>,
        grid_spacing: f64,
    ) -> PyResult<Self> {
        let pos = positions_from_numpy(&positions)?;
        let r = radii.as_slice()?;
        let surface = Surface::from_atoms_gaussian(&pos, r, grid_spacing)?;
        Ok(Self {
            inner: Arc::new(surface),
        })
    }

    /// Write the mesh as a Wavefront `.obj` file (PyMOL `cmd.load`,
    /// VMD, MeshLab, Blender all accept this format).
    fn write_obj(&self, path: &str) -> PyResult<()> {
        self.inner.write_obj(path)?;
        Ok(())
    }

    #[getter]
    fn num_vertices(&self) -> usize {
        self.inner.num_vertices()
    }

    #[getter]
    fn num_faces(&self) -> usize {
        self.inner.num_faces()
    }

    /// Detect whether every charge position sits inside the cavity,
    /// outside it, or whether the set straddles the boundary.
    fn classify_charges(&self, positions: PyReadonlyArray2<'_, f64>) -> PyResult<PyChargeSide> {
        let pos = positions_from_numpy(&positions)?;
        let side = self.inner.classify_charges(&pos)?;
        Ok(side.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "Surface(num_vertices={}, num_faces={})",
            self.inner.num_vertices(),
            self.inner.num_faces()
        )
    }
}

// =========================================================================
// BemSolution
// =========================================================================

#[pyclass(name = "BemSolution", module = "bemtzmann")]
pub struct PyBemSolution {
    // why: holding an Arc lets the surface outlive the Python Surface
    // object in case the user drops it — avoids dangling references.
    surface: Arc<Surface>,
    media: Dielectric,
    side: ChargeSide,
    f: Vec<f64>,
    h: Vec<f64>,
}

impl PyBemSolution {
    fn as_solution(&self) -> BemSolution<'_> {
        BemSolution::from_densities(
            &self.surface,
            self.media,
            self.side,
            self.f.clone(),
            self.h.clone(),
        )
    }
}

#[pymethods]
impl PyBemSolution {
    /// Solve the Juffer BIE for the given surface and source charges.
    ///
    /// `positions` is `(N, 3) float64`, `charges` is `(N,) float64`.
    /// `eps_in`, `eps_out`, `kappa` define the continuum dielectric
    /// (water at 25 °C ≈ 80; protein interior typically 2–4; `kappa`
    /// in Å⁻¹ defaults to 0 = no salt screening). `side=None`
    /// auto-classifies via [`Surface.classify_charges`]; pass an
    /// explicit `ChargeSide` (or skip the ray-cast cost on huge
    /// systems) to override.
    #[staticmethod]
    #[pyo3(signature = (surface, positions, charges, *, eps_in = 4.0, eps_out = 80.0, kappa = 0.0, side = None))]
    fn solve(
        surface: &PySurface,
        positions: PyReadonlyArray2<'_, f64>,
        charges: PyReadonlyArray1<'_, f64>,
        eps_in: f64,
        eps_out: f64,
        kappa: f64,
        side: Option<PyChargeSide>,
    ) -> PyResult<Self> {
        let pos = positions_from_numpy(&positions)?;
        let q = charges.as_slice()?;
        if pos.len() != q.len() {
            return Err(PyValueError::new_err(format!(
                "charge arrays have mismatched lengths: {} positions vs {} values",
                pos.len(),
                q.len()
            )));
        }
        let media = Dielectric::continuum_with_salt(eps_in, eps_out, kappa);
        let side: ChargeSide = match side {
            Some(s) => s.into(),
            None => surface.inner.classify_charges(&pos)?,
        };
        let sol = BemSolution::solve(&surface.inner, media, side, &pos, q)?;
        // Destructure the Rust solution into owned densities so the
        // Python wrapper can live past the Rust borrow.
        let f = sol.surface_potential().to_vec();
        let h = sol.surface_normal_deriv().to_vec();
        Ok(Self {
            surface: Arc::clone(&surface.inner),
            media,
            side,
            f,
            h,
        })
    }

    /// Reaction-field potential `φ_rf` at a single probe point,
    /// reduced units `e/Å`.
    fn reaction_field_at(&self, point: [f64; 3]) -> f64 {
        self.as_solution().reaction_field_at(point)
    }

    /// Batched reaction-field evaluation. Returns a fresh
    /// `(N,) float64` array matching `points.shape[0]`.
    fn reaction_field_at_many<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let pts = positions_from_numpy(&points)?;
        let mut out = vec![0.0_f64; pts.len()];
        self.as_solution().reaction_field_at_many(&pts, &mut out)?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Pairwise reaction-field interaction energy `q_j · φ_rf(r_j)`
    /// for a solve seeded with `(positions, charges)`. Useful for
    /// per-atom decomposition.
    fn interaction_energy(
        &self,
        positions: PyReadonlyArray2<'_, f64>,
        charges: PyReadonlyArray1<'_, f64>,
        i: usize,
        j: usize,
    ) -> PyResult<f64> {
        let pos = positions_from_numpy(&positions)?;
        let q = charges.as_slice()?;
        Ok(self.as_solution().interaction_energy(&pos, q, i, j)?)
    }
}

// =========================================================================
// LinearResponse
// =========================================================================

#[pyclass(name = "LinearResponse", module = "bemtzmann")]
pub struct PyLinearResponse {
    surface: Arc<Surface>,
    media: Dielectric,
    side: ChargeSide,
    sites: Vec<[f64; 3]>,
    // Per-site (f, h) densities, same ordering as `sites`.
    densities: Vec<(Vec<f64>, Vec<f64>)>,
}

impl PyLinearResponse {
    fn as_bem_solutions(&self) -> Vec<BemSolution<'_>> {
        // why: LinearResponse's public evaluators take `&self`, and
        // `LinearResponse` itself holds `Vec<BemSolution<'_>>`. We
        // reconstruct those on demand from the per-site densities
        // since the Python wrapper can't carry the Rust lifetime.
        self.densities
            .iter()
            .map(|(f, h)| {
                BemSolution::from_densities(
                    &self.surface,
                    self.media,
                    self.side,
                    f.clone(),
                    h.clone(),
                )
            })
            .collect()
    }
}

#[pymethods]
impl PyLinearResponse {
    /// Build the basis: one unit-charge solve per site, plus the
    /// symmetrised response matrix. Same media / side conventions as
    /// [`BemSolution.solve`] — `eps_in`, `eps_out`, `kappa` are
    /// keyword-only with defaults; `side=None` auto-classifies the
    /// `sites` against the surface.
    #[staticmethod]
    #[pyo3(signature = (surface, sites, *, eps_in = 4.0, eps_out = 80.0, kappa = 0.0, side = None))]
    fn precompute(
        surface: &PySurface,
        sites: PyReadonlyArray2<'_, f64>,
        eps_in: f64,
        eps_out: f64,
        kappa: f64,
        side: Option<PyChargeSide>,
    ) -> PyResult<Self> {
        let sites_vec = positions_from_numpy(&sites)?;
        let media = Dielectric::continuum_with_salt(eps_in, eps_out, kappa);
        let side: ChargeSide = match side {
            Some(s) => s.into(),
            None => surface.inner.classify_charges(&sites_vec)?,
        };
        let lr = LinearResponse::precompute(&surface.inner, media, side, &sites_vec)?;
        // Drop `lr` to collect per-site densities out of its internal
        // `Vec<BemSolution>`. The public API doesn't expose them
        // directly; re-running basis solves here would duplicate work.
        // Instead we re-solve on a fresh SolveContext through the
        // existing `BemSolution::solve` path, once per site. Same
        // work, cleaner ownership.
        //
        // why: future follow-up could expose an internal iterator
        // that yields `(f, h)` pairs from the LinearResponse state
        // directly without duplicating the solves. For now the
        // precompute cost is dominated by GMRES, not setup, so
        // the O(N_sites) duplication is tolerable in exchange for
        // staying on public Rust APIs.
        drop(lr);
        let mut densities = Vec::with_capacity(sites_vec.len());
        for site in &sites_vec {
            let sol = BemSolution::solve(
                &surface.inner,
                media,
                side,
                std::slice::from_ref(site),
                &[1.0],
            )?;
            densities.push((
                sol.surface_potential().to_vec(),
                sol.surface_normal_deriv().to_vec(),
            ));
        }
        Ok(Self {
            surface: Arc::clone(&surface.inner),
            media,
            side,
            sites: sites_vec,
            densities,
        })
    }

    #[getter]
    fn num_sites(&self) -> usize {
        self.sites.len()
    }

    /// Solvation energy `½ qᵀ G q` for a charge configuration
    /// (reduced units `e²/Å`).
    fn solvation_energy(&self, charges: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let q = charges.as_slice()?;
        if q.len() != self.sites.len() {
            return Err(PyValueError::new_err(format!(
                "charges length {} ≠ num_sites {}",
                q.len(),
                self.sites.len()
            )));
        }
        let bases = self.as_bem_solutions();
        // why: E_solv = ½ Σ_{i,j} q_i q_j · φ_rf_i(r_j). Evaluate the
        // N² terms directly against the per-site basis solutions.
        // LinearResponse's own cached `G` matrix isn't accessible via
        // the public API; reconstructing it here would duplicate its
        // internal symmetrisation logic, so we compute the bilinear
        // form straight.
        let mut acc = 0.0;
        for (i, basis_i) in bases.iter().enumerate() {
            for (j, &q_j) in q.iter().enumerate() {
                acc += q[i] * q_j * basis_i.reaction_field_at(self.sites[j]);
            }
        }
        Ok(0.5 * acc)
    }

    /// Reaction field `(Gq)_j` at every precomputed site for the
    /// given charge configuration. `O(N²)` field evaluations; for
    /// MC hot loops caching the output between accepted moves.
    fn reaction_field_at_sites<'py>(
        &self,
        py: Python<'py>,
        charges: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let q = charges.as_slice()?;
        if q.len() != self.sites.len() {
            return Err(PyValueError::new_err(format!(
                "charges length {} ≠ num_sites {}",
                q.len(),
                self.sites.len()
            )));
        }
        let bases = self.as_bem_solutions();
        let mut out = vec![0.0_f64; self.sites.len()];
        for (j, out_j) in out.iter_mut().enumerate() {
            let r_j = self.sites[j];
            let mut s = 0.0;
            for (i, basis_i) in bases.iter().enumerate() {
                s += q[i] * basis_i.reaction_field_at(r_j);
            }
            *out_j = s;
        }
        Ok(PyArray1::from_vec(py, out))
    }

    /// Reaction-field potential at a single arbitrary probe point
    /// for the given charge configuration.
    fn reaction_field_at(
        &self,
        charges: PyReadonlyArray1<'_, f64>,
        point: [f64; 3],
    ) -> PyResult<f64> {
        let q = charges.as_slice()?;
        if q.len() != self.sites.len() {
            return Err(PyValueError::new_err(format!(
                "charges length {} ≠ num_sites {}",
                q.len(),
                self.sites.len()
            )));
        }
        let bases = self.as_bem_solutions();
        let mut s = 0.0;
        for (i, basis_i) in bases.iter().enumerate() {
            s += q[i] * basis_i.reaction_field_at(point);
        }
        Ok(s)
    }

    /// Batched reaction-field evaluation at many external points.
    fn reaction_field_at_many<'py>(
        &self,
        py: Python<'py>,
        charges: PyReadonlyArray1<'py, f64>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let q = charges.as_slice()?;
        if q.len() != self.sites.len() {
            return Err(PyValueError::new_err(format!(
                "charges length {} ≠ num_sites {}",
                q.len(),
                self.sites.len()
            )));
        }
        let pts = positions_from_numpy(&points)?;
        let bases = self.as_bem_solutions();
        let mut out = vec![0.0_f64; pts.len()];
        for (j, out_j) in out.iter_mut().enumerate() {
            let r = pts[j];
            let mut s = 0.0;
            for (i, basis_i) in bases.iter().enumerate() {
                s += q[i] * basis_i.reaction_field_at(r);
            }
            *out_j = s;
        }
        Ok(PyArray1::from_vec(py, out))
    }
}

// =========================================================================
// Free functions
// =========================================================================

/// Return type for the charge-file readers below: `(positions, charges,
/// radii)`. `positions` is `(N, 3) float64`, `charges` is `(N,) float64`
/// in elementary charges, `radii` is `(N,) float64` in Å — empty when
/// the source format has no radius column (XYZ in 5-column form).
type AtomArrays<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Load atoms from a PQR file. Returns `(positions, charges, radii)`.
/// PQR always carries radii.
#[pyfunction]
fn read_pqr<'py>(py: Python<'py>, path: &str) -> PyResult<AtomArrays<'py>> {
    let atoms = crate::io::read_pqr(path)?;
    Ok((
        positions_to_numpy(py, &atoms.positions),
        PyArray1::from_vec(py, atoms.charges),
        PyArray1::from_vec(py, atoms.radii),
    ))
}

/// Load atoms from an XYZ-style file. Returns `(positions, charges,
/// radii)`. `radii` is empty (zero-length array) when the file uses the
/// 5-column form without radii.
#[pyfunction]
fn read_xyz<'py>(py: Python<'py>, path: &str) -> PyResult<AtomArrays<'py>> {
    let atoms = crate::io::read_xyz(path)?;
    Ok((
        positions_to_numpy(py, &atoms.positions),
        PyArray1::from_vec(py, atoms.charges),
        PyArray1::from_vec(py, atoms.radii),
    ))
}

/// Convert a reduced-units energy (`e²/Å`) to kJ/mol.
#[pyfunction(name = "to_kJ_per_mol")]
fn to_kj_per_mol(energy: f64) -> f64 {
    crate::units::to_kJ_per_mol(energy)
}

/// Convert a reduced-units energy (`e²/Å`) to dimensionless `kT` at
/// the given temperature (K).
#[pyfunction]
fn to_kt(energy: f64, temperature_k: f64) -> f64 {
    crate::units::to_kt(energy, temperature_k)
}

// =========================================================================
// Module entry point
// =========================================================================

#[pymodule]
#[pyo3(name = "_bemtzmann")]
fn bemtzmann(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChargeSide>()?;
    m.add_class::<PyDielectric>()?;
    m.add_class::<PySurface>()?;
    m.add_class::<PyBemSolution>()?;
    m.add_class::<PyLinearResponse>()?;
    m.add_function(wrap_pyfunction!(read_pqr, m)?)?;
    m.add_function(wrap_pyfunction!(read_xyz, m)?)?;
    m.add_function(wrap_pyfunction!(to_kj_per_mol, m)?)?;
    m.add_function(wrap_pyfunction!(to_kt, m)?)?;
    Ok(())
}
