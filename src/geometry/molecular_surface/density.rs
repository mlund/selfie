//! Gaussian molecular-surface density evaluator.
//!
//! Defines
//! ρ(r) = Σᵢ exp(d · (1 − |r − rᵢ|² / Rᵢ²))
//! and its analytical gradient. Atoms whose contribution falls below a small
//! ε at the query point are skipped via a KD-tree radius query.
//!
//! The mesh is the isolevel ρ = ρ₀; the gradient direction `−∇ρ / ‖∇ρ‖` is the
//! outward normal (density is high inside the molecule, low outside).

use crate::error::{Error, Result};
#[cfg(test)]
use glam::DVec3;

/// Map a kd-tree failure into a structured selfie error. The kd-tree only
/// fails on `WrongDimension` (statically impossible — we always use 3D) or
/// `NonFiniteCoordinate` (caller-side bug — `validate_inputs` should have
/// caught this). Either way, surface as `NonManifoldMesh` rather than
/// panicking.
fn kdtree_err(e: kdtree::ErrorKind) -> Error {
    Error::NonManifoldMesh {
        reason: format!("kd-tree internal failure: {e:?}"),
    }
}

/// Returns the squared radius beyond which a sphere of decay `decay` and
/// radius `radius` contributes less than `eps` to ρ.
///
/// Solving exp(d·(1 − r²/R²)) < ε gives r² > R² · (1 − ln(ε)/d). Used both to
/// size grid padding (boundary cells must be far enough from atoms that
/// ρ ≪ ρ₀) and as the per-atom KD-tree query radius during ρ evaluation.
pub(super) fn cutoff_radius_sq(radius: f64, decay: f64, eps: f64) -> f64 {
    debug_assert!(radius > 0.0 && decay > 0.0 && (0.0..1.0).contains(&eps));
    radius * radius * (1.0 - eps.ln() / decay)
}

/// Build a flat-array KD-tree over atom positions. The stored value at each
/// point is the atom index, so cutoff queries return the indices we need to
/// look up the corresponding radius.
pub(super) fn build_atom_tree(
    positions: &[[f64; 3]],
) -> Result<kdtree::KdTree<f64, u32, [f64; 3]>> {
    let mut tree: kdtree::KdTree<f64, u32, [f64; 3]> = kdtree::KdTree::new(3);
    for (i, p) in positions.iter().enumerate() {
        tree.add(*p, i as u32).map_err(kdtree_err)?;
    }
    Ok(tree)
}

/// Evaluate ρ at a single point, skipping atoms beyond the system-wide cutoff.
///
/// `query_cutoff_sq` is the *largest* per-atom cutoff in the system (we'd
/// rather pay for a few small extra contributions than tune per-atom). Atoms
/// outside that radius contribute < `cutoff_eps` per the cutoff math and are
/// dropped via the KD-tree's `within` query.
pub(super) fn density_at(
    point: [f64; 3],
    radii_sq: &[f64],
    decay: f64,
    query_cutoff_sq: f64,
    tree: &kdtree::KdTree<f64, u32, [f64; 3]>,
) -> Result<f64> {
    let hits = tree
        .within(
            &point,
            query_cutoff_sq,
            &kdtree::distance::squared_euclidean,
        )
        .map_err(kdtree_err)?;
    let mut rho = 0.0_f64;
    for (dist_sq, &idx) in hits {
        let r_sq = radii_sq[idx as usize];
        rho += (decay * (1.0 - dist_sq / r_sq)).exp();
    }
    Ok(rho)
}

/// Analytical gradient ∇ρ at a point.
///
/// Closed form:
///   ∇ρ(r) = −(2d) · Σᵢ φᵢ(r) · (r − rᵢ) / Rᵢ²
/// where φᵢ is the i-th Gaussian. Outward normal at any vertex on the surface
/// is `−∇ρ / ‖∇ρ‖`. Used for unit tests and (eventually) vertex-normal
/// shading; the production face normals come from `panel::FaceGeoms::compute`.
#[cfg(test)]
pub(super) fn gradient_at(
    point: [f64; 3],
    positions: &[[f64; 3]],
    radii: &[f64],
    decay: f64,
) -> [f64; 3] {
    let p = DVec3::from(point);
    let mut grad = DVec3::ZERO;
    for (pos, &radius) in positions.iter().zip(radii) {
        let q = DVec3::from(*pos);
        let r_sq = radius * radius;
        let diff = p - q;
        let dist_sq = diff.length_squared();
        let phi = (decay * (1.0 - dist_sq / r_sq)).exp();
        grad += -(2.0 * decay) * phi * diff / r_sq;
    }
    grad.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: f64 = 1.0;
    const EPS: f64 = 1e-6;

    #[test]
    fn cutoff_radius_matches_closed_form() {
        let r = 1.7;
        let r_cut_sq = cutoff_radius_sq(r, D, EPS);
        let r_cut = r_cut_sq.sqrt();
        // r_cut = R · √(1 − ln(ε)/d). For d=1, ε=1e-6: factor ≈ 3.85.
        let expected = r * (1.0 - EPS.ln() / D).sqrt();
        assert!((r_cut - expected).abs() < 1e-12, "{r_cut} vs {expected}");
    }

    #[test]
    fn density_at_atom_centre_is_e_to_d() {
        // ρ(rᵢ) = exp(d · 1) for one isolated atom.
        let positions = [[0.0, 0.0, 0.0]];
        let radii = [1.5_f64];
        let radii_sq = [radii[0] * radii[0]];
        let tree = build_atom_tree(&positions).unwrap();
        let cutoff_sq = cutoff_radius_sq(radii[0], D, EPS);
        let rho = density_at([0.0; 3], &radii_sq, D, cutoff_sq, &tree).unwrap();
        assert!((rho - D.exp()).abs() < 1e-12, "ρ(centre) = {rho}");
    }

    #[test]
    fn density_at_sphere_surface_is_one_for_isolated_atom() {
        // |r − rᵢ| = Rᵢ ⇒ ρ = exp(0) = 1.
        let positions = [[0.0, 0.0, 0.0]];
        let radii = [2.0_f64];
        let radii_sq = [radii[0] * radii[0]];
        let tree = build_atom_tree(&positions).unwrap();
        let cutoff_sq = cutoff_radius_sq(radii[0], D, EPS);
        let rho = density_at([2.0, 0.0, 0.0], &radii_sq, D, cutoff_sq, &tree).unwrap();
        assert!((rho - 1.0).abs() < 1e-12, "ρ(surface) = {rho}");
    }

    #[test]
    fn gradient_matches_finite_difference() {
        // Two-atom system; check ∇ρ vs. central differences of ρ.
        let positions = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let radii = [1.5_f64, 1.5];
        let radii_sq = [radii[0] * radii[0], radii[1] * radii[1]];
        let tree = build_atom_tree(&positions).unwrap();
        let cutoff_sq = cutoff_radius_sq(radii[0].max(radii[1]), D, EPS);
        let p = [1.0, 0.5, -0.3];

        let h = 1e-5;
        let mut numerical = [0.0_f64; 3];
        for axis in 0..3 {
            let mut p_plus = p;
            let mut p_minus = p;
            p_plus[axis] += h;
            p_minus[axis] -= h;
            let f_plus = density_at(p_plus, &radii_sq, D, cutoff_sq, &tree).unwrap();
            let f_minus = density_at(p_minus, &radii_sq, D, cutoff_sq, &tree).unwrap();
            numerical[axis] = (f_plus - f_minus) / (2.0 * h);
        }
        let analytical = gradient_at(p, &positions, &radii, D);
        for (axis, (n, a)) in numerical.iter().zip(analytical).enumerate() {
            assert!(
                (n - a).abs() < 1e-7,
                "axis {axis}: FD {n} vs analytical {a}"
            );
        }
    }

    #[test]
    fn density_skips_distant_atoms() {
        // An atom 100 Å away should contribute ~0 (< cutoff_eps) and be
        // excluded from the KD-tree query. The query result must equal the
        // single-atom density without it.
        let positions = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]];
        let radii = [1.5_f64, 1.5];
        let radii_sq = [radii[0] * radii[0], radii[1] * radii[1]];
        let tree = build_atom_tree(&positions).unwrap();
        let cutoff_sq = cutoff_radius_sq(radii[0].max(radii[1]), D, EPS);
        let rho = density_at([0.0; 3], &radii_sq, D, cutoff_sq, &tree).unwrap();
        assert!((rho - D.exp()).abs() < 1e-12, "{rho}");
    }
}
