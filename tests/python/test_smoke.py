"""End-to-end smoke coverage for the selfie Python bindings.

Validates that the native module loads cleanly, numpy interop works at
construction and evaluation time, the solver recovers a known closed-
form answer (Born), and that caller misuse surfaces as Python
exceptions rather than panics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import selfie as s

LYSOZYME_PQR = Path(__file__).resolve().parent.parent / "data" / "pygbe_lys" / "built_parse.pqr"


def test_icosphere_shape():
    surf = s.Surface.icosphere(10.0, 3)
    # subdiv = 3 → 20 · (3 + 1)² = 320 triangles, 162 vertices.
    assert surf.num_faces == 320
    assert surf.num_vertices == 162


def test_born_reaction_field_at_center():
    # Single unit charge at the origin of a radius-10 Å sphere,
    # ε_in = 2, ε_out = 80. Closed form: φ_rf(0) = q/a · (1/ε_out − 1/ε_in).
    surf = s.Surface.icosphere(10.0, 3)
    media = s.Dielectric(eps_in=2.0, eps_out=80.0)
    pos = np.array([[0.0, 0.0, 0.0]])
    q = np.array([1.0])
    sol = s.BemSolution.solve(surf, media, s.ChargeSide.Interior, pos, q)
    phi = sol.reaction_field_at((0.0, 0.0, 0.0))
    expected = (1.0 / 80.0 - 1.0 / 2.0) / 10.0
    # 2 % tolerance at subdiv = 3 (centroid collocation is O(h²)).
    assert abs(phi - expected) / abs(expected) < 0.02


def test_linear_response_quadratic_scaling():
    # Solvation energy must scale as λ² under uniform charge rescaling.
    surf = s.Surface.icosphere(10.0, 3)
    media = s.Dielectric(eps_in=2.0, eps_out=80.0)
    sites = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    basis = s.LinearResponse.precompute(surf, media, s.ChargeSide.Interior, sites)
    assert basis.num_sites == 3

    q = np.array([1.0, -0.5, 0.3])
    e_base = basis.solvation_energy(q)
    for lam in (0.5, 2.0, 3.7):
        e_scaled = basis.solvation_energy(lam * q)
        assert abs(e_scaled - lam * lam * e_base) / abs(e_base) < 1e-12


def test_read_pqr_lysozyme_shapes():
    if not LYSOZYME_PQR.exists():
        pytest.skip("lysozyme fixture not present")
    positions, charges = s.read_pqr(str(LYSOZYME_PQR))
    # built_parse.pqr ships with 1323 atoms.
    assert positions.shape == (1323, 3)
    assert charges.shape == (1323,)
    assert positions.dtype == np.float64
    assert charges.dtype == np.float64


def test_charge_length_mismatch_raises_value_error():
    surf = s.Surface.icosphere(10.0, 3)
    media = s.Dielectric(eps_in=2.0, eps_out=80.0)
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    q = np.array([1.0])  # length mismatch vs 2 positions
    with pytest.raises(ValueError, match="mismatch|positions"):
        s.BemSolution.solve(surf, media, s.ChargeSide.Interior, pos, q)
