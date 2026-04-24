# SelFie

A boundary-element solver for the linearized Poisson–Boltzmann equation,
written in Rust. Computes the electrostatic response of a protein (or any
closed dielectric object) to a set of point charges, in a continuum-solvent
model with optional salt screening.

## Quick start

Not yet on PyPI; install straight from the repository:

```sh
uv add "selfie @ git+https://github.com/mlund/selfie.git"
# or, with plain pip:
pip install "git+https://github.com/mlund/selfie.git"
```

The installer builds the native extension via `maturin`; no Rust toolchain
on the user side is needed at install time if a prebuilt wheel is
available, otherwise `cargo` is pulled in automatically.

Minimal working example — a single unit charge at the centre of a
dielectric sphere, compared against the Born closed form:

```python
import numpy as np
import selfie as s

surface = s.Surface.icosphere(radius=10.0, subdivisions=5)
media = s.Dielectric(eps_in=2.0, eps_out=80.0)

positions = np.array([[0.0, 0.0, 0.0]])
charges = np.array([1.0])
sol = s.BemSolution.solve(surface, media, s.ChargeSide.Interior, positions, charges)

u_born = 0.5 * charges[0] * sol.reaction_field_at((0.0, 0.0, 0.0))
print(f"Born solvation energy: {s.to_kJ_per_mol(u_born):.2f} kJ/mol")
```

## Usage

### Solving on a pre-built mesh

`Surface.from_msms` reads the standard MSMS `.vert` / `.face` pair;
`read_pqr` loads atom charges. `classify_charges` detects which side
of the dielectric boundary the atoms live on, so you don't have to
hard-code it.

```python
import selfie as s

surface = s.Surface.from_msms("Lys1.vert", "Lys1.face")
positions, charges = s.read_pqr("built_parse.pqr")
side = surface.classify_charges(positions)

media = s.Dielectric(eps_in=4.0, eps_out=80.0, kappa=0.125)  # physiological salt
sol = s.BemSolution.solve(surface, media, side, positions, charges)
```

### Solvation energy of a protein

```python
import numpy as np
import selfie as s

# E_solv = ½ Σ_j q_j · φ_rf(r_j), summed over all source charges.
phi = sol.reaction_field_at_many(positions)
e_solv_reduced = 0.5 * np.dot(charges, phi)
print(f"E_solv = {s.to_kJ_per_mol(e_solv_reduced):.2f} kJ/mol")
```

### Many charge configurations over the same mesh

When you need to evaluate the solver repeatedly over the *same* mesh
with *different* charges (binding-energy scans, pKa shifts, sensitivity
studies), precompute the linear-response basis once and turn every
downstream query into dense linear algebra:

```python
basis = s.LinearResponse.precompute(surface, media, side, sites=positions)

q1 = charges
q2 = charges.copy()
q2[0] += 1.0   # flip one site's protonation

e1 = basis.solvation_energy(q1)
e2 = basis.solvation_energy(q2)
print(f"ΔE_solv = {s.to_kJ_per_mol(e2 - e1):.2f} kJ/mol")
```

The precompute cost is `N_sites` full BEM solves; every later
`solvation_energy` call is `O(N_sites²)` flops — microseconds even at
hundreds of sites.

### Reaction field on a grid

For visualisation or post-hoc analysis, sample the reaction field at
arbitrary probe points:

```python
import numpy as np

probe = np.array([[0.0, 0.0, z] for z in np.linspace(-10, 10, 21)])
phi_rf = sol.reaction_field_at_many(probe)
```

## Overview

Given:
- a triangulated surface enclosing a low-dielectric cavity (e.g. a protein
  solvent-excluded surface),
- an interior and exterior dielectric constant (`ε_in`, `ε_out`),
- an optional Debye–Hückel inverse length `κ` for the solvent,
- a set of point charges living on one side of the surface (typically
  inside the protein),

selfie returns:
- the surface potential and its normal derivative at each triangle's
  centroid,
- the **reaction-field potential** `φ_rf(r)` at any probe point — the
  polarisation response of the dielectric/ionic environment, free of the
  direct Coulomb contribution of the source charges,
- the **solvation energy** `E_solv = ½ ∫ ρ · φ_rf dV` as a derived quantity.

For workflows that evaluate many charge configurations over the same mesh
(pKa shifts, binding-energy scans, MC pH titration), a precomputed
linear-response basis reduces every further query to O(N_sites²) linear
algebra — the boundary-integral equation is linear in the charges, so
one upfront precompute (one BEM solve per site, with unit charge)
replaces the entire trajectory of naïve per-trial solves.

Target meshes are SES-like triangulations of real biomolecules —
lysozyme at 14 k faces solves in ≈ 1.6 s on a laptop, matching
[pyGBe](https://github.com/pygbe/pygbe)'s reference E_solv to 0.1 %.

selfie reads standard community mesh + charge formats: MSMS `.vert` /
`.face` for geometry and PQR for charges.

## Theory

### Continuum-electrostatics model

The protein interior is treated as a homogeneous low-dielectric cavity
`Ω⁻` with permittivity `ε_in`; the solvent exterior `Ω⁺` has `ε_out` and
may contain a screening ionic atmosphere characterised by the inverse
Debye length `κ`. The potential `φ(r)` satisfies

```
∇²φ = −ρ/ε_in                 inside  Ω⁻   (point-charge Poisson)
(∇² − κ²) φ = 0               outside Ω⁺   (linearised Poisson–Boltzmann)
φ, ε ∂_n φ continuous         across the boundary Γ
```

Setting `κ = 0` reduces the exterior to pure Laplace (no salt).

### Boundary-integral reformulation (Juffer 1991)

Taking `φ` and `h = ∂_n φ_out` as the unknowns on the surface, the
problem reduces to the Juffer derivative block system

```
[ ½I + K'_0        −(ε_out/ε_in) K_0 ] [ φ ]   [ φ_source_in ]
[ ½I − K'_κ              K_κ         ] [ h ] = [ φ_source_out ]
```

where `K_0`, `K'_0`, `K_κ`, `K'_κ` are panel-integrated single- and
double-layer operators for the Laplace and Yukawa Green's functions.
Only one right-hand side block is nonzero, set by which side of the
boundary the source charges inhabit. Once `(φ, h)` is known, the
reaction-field potential at any external point follows from Green's
third identity and is evaluated by integration over Γ.

### Solvation energy and linearity in charges

The boundary operator depends on geometry and dielectric only; the
right-hand side is a linear sum of per-site terms in the source charges.
The solution `(φ, h)` and therefore the reaction field are linear in
the charge vector `q`, and the solvation energy takes the bilinear form

```
E_solv(q) = ½ qᵀ G q           with   G_ij = φ_rf_i(r_j)
```

where `φ_rf_i(r_j)` is the reaction field at site `j` induced by a unit
source at site `i`. Pre-computing the `G` matrix once converts any
downstream charge-assignment sweep into pure linear algebra.

### Units

All public quantities are in reduced electrostatic units:

| quantity  | unit   |
|-----------|--------|
| length    | Å      |
| charge    | e (elementary charge) |
| potential | e / Å  |
| energy    | e² / Å |
| inverse Debye length | Å⁻¹ |

A helper is provided to convert energies to kJ/mol.

## Validation

selfie is validated against closed-form analytical solutions on a
dielectric sphere and against an independent reference implementation on
a real protein mesh.

### Analytical references (sphere geometry)

| Reference | Geometry | Tolerance at `n_t = 1280` |
|---|---|---|
| **Born** (`κ = 0`) | single charge at sphere centre | `< 0.5 %` |
| **Debye–Hückel** (`κ > 0`) | single charge at sphere centre | `< 0.5 %` |
| **Kirkwood interior** (`κ = 0`) | two charges inside the cavity | `< 1 %` |
| **Kirkwood interior + salt** (`κ > 0`) | axisymmetric charges on z-axis with screened solvent | `< 2 %` |
| **Kirkwood exterior** (`κ = 0`) | charges outside the cavity | `< 1 %` |
| **Kirkwood exterior + salt** (`κ > 0`) | screened solvent, exterior charges | `< 1 %` |
| **Onsager** | point dipole at the cavity centre | `< 1 %` |

Each reference closes a face of the `{inside, outside} × {κ = 0, κ > 0}`
coverage grid.

### Cross-validation against pyGBe

On the **lysozyme** SES mesh (`Lys1`, 14 398 faces, 1 323 atoms, ε_in = 4,
ε_out = 80, κ = 0.125 Å⁻¹) selfie matches pyGBe's reference
`E_solv = −573.90 kcal/mol` to **0.10 %**.

Sphere-mesh runs against the pyGBe validation fixture
(R = 4 Å, interior unit charge, physiological salt) match the same
analytical reference pyGBe does, to `< 2 %`.

### Self-consistency gates

- **Reciprocity** of the discretised operator: `φ_rf(r_j; r_i)` and
  `φ_rf(r_i; r_j)` must agree for `i ≠ j`, exercised on both sides of
  the dielectric boundary and with/without salt.
- **No-contrast** sanity: when `ε_in = ε_out` the reaction field must
  collapse to zero.
- **Linear-response consistency**: the bilinear energy reconstructed
  from the basis matches a direct BEM solve with the same charges.
- **Physical sign**: the solvent-stabilising sign of the reaction field
  is enforced on like-charge interior pairs.

### Numerical convergence

The solver uses centroid collocation, which is O(h²) in panel size.
Convergence rates are checked automatically in the salt-bridge tests: a
4× mesh refinement must reduce the error by at least 3× to pass.

### Running the tests

```sh
cargo test --release --features validation
```

runs the full ~90-test suite including all analytical-reference matches.
The end-to-end lysozyme solve is gated behind `--ignored`:

```sh
cargo test --release --features validation --test pygbe_lys_mesh -- --ignored
```
