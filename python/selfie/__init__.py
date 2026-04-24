"""selfie — boundary-element solver for the linearised Poisson–Boltzmann equation.

The entire public API is re-exported from the native ``_selfie`` extension
module. See ``help(selfie.Surface)`` and friends for detailed signatures;
the project README shows end-to-end usage.
"""

from ._selfie import (
    BemSolution,
    ChargeSide,
    Dielectric,
    LinearResponse,
    Surface,
    read_pqr,
    read_xyz,
    to_kJ_per_mol,
    to_kt,
)

__all__ = [
    "BemSolution",
    "ChargeSide",
    "Dielectric",
    "LinearResponse",
    "Surface",
    "read_pqr",
    "read_xyz",
    "to_kJ_per_mol",
    "to_kt",
]
