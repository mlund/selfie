# Test data

## `pygbe_sphere/`

Copied from [pygbe](https://github.com/pygbe/pygbe)'s `examples/sphere/`
(BSD-3 license). Used by `tests/pygbe_sphere_mesh.rs` for cross-BEM
validation against pygbe's stored regression result of
`E_solv = −13.458119761457832 kcal/mol` (from `pygbe/tests/sphere.pickle`).

- `sphere500_R4.vert` / `.face`: MSMS-format triangulation of a 4 Å
  sphere (nominal 500 triangles; actually 512).
- `offcenter_R2.pqr`: single unit charge at `|r| = 2 Å`, i.e. 2 Å beneath
  the sphere surface.

## `pygbe_lys/`

Copied from [pygbe](https://github.com/pygbe/pygbe)'s `examples/lys/`
(BSD-3 license). Lysozyme (PDB 1LYZ or similar) molecular surface and
point charges, for running real-protein tests against pygbe's stored
regression value of `E_solv = −581.2948612575786 kcal/mol` (from
`pygbe/tests/lysozyme.pickle`).

**Important caveat**: pygbe's stored number uses a full 5-surface setup
(outer Stern layer + protein/solvent interface + 3 internal cavities).
Our solver currently handles only a single closed surface, so we exercise
just `Lys1.{vert,face}` (the protein surface) and expect our energy to
differ from pygbe's by a nontrivial amount. The fixture is here to
verify our I/O + manifold + side-classification pipeline handles a real
protein mesh — not to reproduce the exact kcal/mol.

- `Lys1.vert` / `Lys1.face`: MSMS-format SES surface of the protein
  (7,200 vertices, 14,398 faces). From pygbe `examples/lys/geometry/`.
- `lys1_charges.pqr`: charges used by pygbe's full 5-surface `lys`
  example (~1,100 atoms). From pygbe `examples/lys/`.
- `built_parse.pqr`: the charges used by pygbe's **single-surface**
  convergence test `lys_single_1.config` (1,323 atoms, slightly
  different parameterisation of the same protein). From pygbe
  `tests/convergence_tests/input_files/`. This is the file our
  `tests/pygbe_lys_mesh.rs` uses.
