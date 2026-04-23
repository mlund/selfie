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
