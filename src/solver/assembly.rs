//! Build the Juffer derivative-BIE block system.
//!
//! ```text
//! [ ½I + K₀'     −(ε_out/ε_in) K₀ ] [ f ]   [     0        ]
//! [ ½I − K_κ'            K_κ      ] [ h ] = [ φ_yukawa     ]
//! ```
//!
//! - Top row comes from Green's 3rd identity applied to the Laplacian
//!   interior (κ-independent; always G_0).
//! - Bottom row uses the exterior Green's function G_κ and the
//!   corresponding screened Coulomb source
//!   `φ_yukawa(c_a) = Σ_i q_i · exp(−κ|c_a − r_i|) / (ε_out · |c_a − r_i|)`.
//!   For `κ = 0` this reduces to the original Laplace BIE.
//!
//! Reference: Juffer et al., *J. Comput. Phys.* 97 (1991) 144–171,
//! https://doi.org/10.1016/0021-9991(91)90043-K.

use crate::geometry::Surface;
use crate::solver::panel_integrals;
use crate::units::Dielectric;
use faer::Mat;
use glam::DVec3;
use rayon::prelude::*;

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
    let kappa = media.kappa;

    // why: we build the matrix in a flat row-major `Vec<f64>` because the
    // outer a-loop is embarrassingly parallel at the row level (each `a`
    // writes a disjoint top row and bottom row) while `faer::Mat` is
    // column-major and doesn't expose a clean parallel-row mutation API.
    // Converting to `faer::Mat` once at the end costs an O(size²) memcpy,
    // negligible next to the O(size²) kernel work.
    let mut flat = vec![0.0_f64; size * size];
    let (flat_top, flat_bot) = flat.split_at_mut(n * size);

    flat_top
        .par_chunks_mut(size)
        .zip(flat_bot.par_chunks_mut(size))
        .enumerate()
        .for_each(|(a, (top_row, bot_row))| {
            let ca = geom.centroids[a];
            for b in 0..n {
                let nb = geom.normals[b];
                let ab = geom.areas[b];
                let tri = geom.tris[b];

                // why: top block needs G_0 kernels (Laplace interior),
                // bottom block needs G_κ kernels (screened PB exterior).
                // Self-panel K₀' and K_κ' PVs both vanish for flat
                // centroid-collocation — same planar-displacement argument.
                let (k0, k0p, kk, kkp) = if a == b {
                    (
                        panel_integrals::k_self(tri, ca, 0.0),
                        0.0,
                        panel_integrals::k_self(tri, ca, kappa),
                        0.0,
                    )
                } else {
                    (
                        panel_integrals::k_off(ca, tri, ab, 0.0),
                        panel_integrals::k_prime_off(ca, tri, nb, ab, 0.0),
                        panel_integrals::k_off(ca, tri, ab, kappa),
                        panel_integrals::k_prime_off(ca, tri, nb, ab, kappa),
                    )
                };

                let half_i = if a == b { 0.5 } else { 0.0 };
                top_row[b] = half_i + k0p;
                top_row[n + b] = -eps_ratio * k0;
                bot_row[b] = half_i - kkp;
                bot_row[n + b] = kk;
            }
        });

    // Flat row-major → faer column-major.
    let mat = Mat::<f64>::from_fn(size, size, |i, j| flat[i * size + j]);

    // RHS: [0; φ_yukawa]. Tiny loop, not worth parallelising.
    let mut rhs = vec![0.0_f64; size];
    for (a, out) in rhs.iter_mut().skip(n).enumerate() {
        let ca = geom.centroids[a];
        let mut phi = 0.0;
        for (pos, &q) in charge_positions.iter().zip(charge_values.iter()) {
            let dist = (ca - DVec3::from(*pos)).length();
            // why: screened-Coulomb (Yukawa) source in reduced units —
            // prefactor 1/ε_out, no 4π or k_e. κ = 0 ⇒ plain Coulomb.
            phi += q * (-kappa * dist).exp() / (media.eps_out * dist);
        }
        *out = phi;
    }

    (mat, rhs)
}
