//! Build the Juffer derivative-BIE block system for κ = 0:
//!
//! ```text
//! [ ½I + K₀'     −(ε_out/ε_in) K₀ ] [ f ]   [    0    ]
//! [ ½I − K₀'            K₀        ] [ h ] = [ φ_coul  ]
//! ```
//!
//! `f` = surface potential at each face centroid, `h` = outward normal
//! derivative of φ_out, `φ_coul(c_a) = Σ_i q_i / (ε_out · |c_a − r_i|)`
//! in reduced units. Reference: Juffer et al., *J. Comput. Phys.* 97
//! (1991) 144–171, https://doi.org/10.1016/0021-9991(91)90043-K.

use crate::geometry::Surface;
use crate::solver::panel_integrals;
use crate::units::Dielectric;
use faer::Mat;
use glam::DVec3;

pub(crate) fn build_block_system(
    surface: &Surface,
    media: Dielectric,
    charge_positions: &[[f64; 3]],
    charge_values: &[f64],
) -> (Mat<f64>, Vec<f64>) {
    let geom = surface.geom_internal();
    let n = geom.len();
    let size = 2 * n;

    let eps_ratio = media.eps_out / media.eps_in;

    let mut mat = Mat::<f64>::zeros(size, size);

    // why: we walk rows = observers a, columns = sources b. Diagonal case
    // (a == b) uses the WRG closed form for K₀ and exact 0 for K₀' PV; the
    // ½I jump is added as a constant to the diagonal of both blocks that
    // carry it (top-left gets +½I, bottom-left gets +½I; K₀' blocks get the
    // integrated K₀' values, with 0 on the self-panel).
    for a in 0..n {
        let ca = geom.centroids[a];
        for b in 0..n {
            let nb = geom.normals[b];
            let ab = geom.areas[b];
            let tri = geom.tris[b];

            let (k0, k0p) = if a == b {
                // Self-panel. K₀: Wilton–Rao–Glisson closed form.
                // K₀': PV vanishes for flat panel at its own centroid.
                (panel_integrals::k0_self_wrg(tri, ca), 0.0)
            } else {
                (
                    panel_integrals::k0_off(ca, tri, ab),
                    panel_integrals::k0_prime_off(ca, tri, nb, ab),
                )
            };

            // Top-left: ½I + K₀'
            // Top-right: −(ε_out/ε_in) K₀
            // Bottom-left: ½I − K₀'
            // Bottom-right: K₀
            let half_i = if a == b { 0.5 } else { 0.0 };
            mat[(a, b)] = half_i + k0p;
            mat[(a, n + b)] = -eps_ratio * k0;
            mat[(n + a, b)] = half_i - k0p;
            mat[(n + a, n + b)] = k0;
        }
    }

    // RHS: [0; φ_coul].
    let mut rhs = vec![0.0_f64; size];
    for a in 0..n {
        let ca = geom.centroids[a];
        let mut phi = 0.0;
        for (pos, &q) in charge_positions.iter().zip(charge_values.iter()) {
            let r = DVec3::from(*pos);
            let dist = (ca - r).length();
            // why: reduced-units Coulomb — prefactor is 1/ε_out, no 4π or k_e.
            // This matches the kernel convention (bem_pb_plan.md §0).
            phi += q / (media.eps_out * dist);
        }
        rhs[n + a] = phi;
    }

    (mat, rhs)
}
