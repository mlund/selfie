//! Kirkwood 1934 + Debye-Hückel analytical solvation energy for point
//! charges inside a dielectric sphere with a salty exterior.
//!
//! Restricted to **charges on the z-axis** (axisymmetric — the m = 0
//! spherical-harmonic sector only). This is enough to validate the
//! interior+salt BEM pipeline against a multipole reference; the full
//! 3D version needs associated Legendre polynomials and is not ported
//! here.
//!
//! Ported from pygbe's `an_P` analytical
//! (<https://github.com/pygbe/pygbe>, `pygbe/util/an_solution.py`).
//!
//! Supports an optional Stern layer of radius `r_stern ≥ r_sphere`
//! (ions excluded from the shell). Set `r_stern = r_sphere` for no
//! Stern layer — the case our single-surface BEM models directly.
//!
//! All in reduced units (charge `e`, distance Å, energy `e²/Å`).

/// pygbe's polynomial `K_n(x)` from Kirkwood 1934 eq 4 — not the
/// modified spherical Bessel `k_n`, a related polynomial factor that
/// appears in the salt-screening correction `C1·C2`:
/// `K_n(x) = Σ_{s=0}^n 2^s · n!·(2n−s)! / (s!·(2n)!·(n−s)!) · x^s`.
fn k_poly(x: f64, n: usize) -> f64 {
    let n_fact = factorial(n);
    let n_fact2 = factorial(2 * n);
    (0..=n)
        .map(|s| {
            let coeff = (1u64 << s) as f64 * n_fact * factorial(2 * n - s)
                / (factorial(s) * n_fact2 * factorial(n - s));
            coeff * x.powi(s as i32)
        })
        .sum()
}

fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, k| acc * k as f64)
}

/// Total reaction-field solvation energy of point charges sitting on the
/// z-axis inside a dielectric sphere of radius `r_sphere`, with salty
/// exterior (Debye inverse length `kappa`) and optional Stern layer
/// radius `r_stern` (use `r_sphere` for no Stern layer).
///
/// Input: `charges_on_z = &[(q_k, z_k)]`. All `z_k` must satisfy
/// `|z_k| < r_sphere`. Output is the total solvation energy
/// `(1/2) · Σ_K q_K · φ_rf(r_K)` in reduced `e²/Å`, summed over the
/// reaction-field contributions at each source location.
pub fn solvation_energy_z_axis(
    charges_on_z: &[(f64, f64)],
    eps_in: f64,
    eps_out: f64,
    r_sphere: f64,
    r_stern: f64,
    kappa: f64,
    n_max: usize,
) -> f64 {
    debug_assert!(r_stern >= r_sphere, "Stern radius must be ≥ sphere radius");
    debug_assert!(
        charges_on_z.iter().all(|&(_, z)| z.abs() < r_sphere),
        "all charges must lie strictly inside the sphere"
    );
    let a = r_stern;
    let r = r_sphere;
    let ka = kappa * a;

    let mut phi = vec![0.0_f64; charges_on_z.len()];
    for n in 0..=n_max {
        let nf = n as f64;
        // why: for z-axis charges, r_k^n · P_n(cos θ_k) simplifies to
        // z_k^n (signed), since cos θ_k = sign(z_k) and P_n(±1) = (±1)^n.
        let e_n: f64 = charges_on_z
            .iter()
            .map(|&(q, z)| q * z.powi(n as i32))
            .sum();

        let b_n = if n == 0 {
            // Born + Debye-Hückel monopole term.
            e_n * (1.0 / eps_out - 1.0 / eps_in) / r - e_n * kappa / (eps_out * (1.0 + ka))
        } else {
            // Kirkwood-inside κ = 0 dipole-and-higher term …
            let kirkwood_term =
                (eps_in - eps_out) * (nf + 1.0) / (eps_in * nf + eps_out * (nf + 1.0)) * e_n
                    / (eps_in * r.powi(2 * n as i32 + 1));

            // … minus the salt correction `C1·C2` that captures Yukawa
            // screening of the exterior response.
            let c1 = e_n / (eps_out * a.powi(2 * n as i32 + 1)) * (2.0 * nf + 1.0)
                / (2.0 * nf - 1.0)
                * (eps_out / ((nf + 1.0) * eps_out + nf * eps_in)).powi(2);
            let k_nm1 = k_poly(ka, n - 1);
            let k_np1 = k_poly(ka, n + 1);
            let denom = k_np1
                + nf * (eps_out - eps_in) / ((nf + 1.0) * eps_out + nf * eps_in)
                    * (r / a).powi(2 * n as i32 + 1)
                    * ka.powi(2)
                    * k_nm1
                    / ((2.0 * nf - 1.0) * (2.0 * nf + 1.0));
            let c2 = ka.powi(2) * k_nm1 / denom;

            kirkwood_term - c1 * c2
        };

        for (phi_k, &(_, z_k)) in phi.iter_mut().zip(charges_on_z) {
            *phi_k += b_n * z_k.powi(n as i32);
        }
    }

    0.5 * charges_on_z
        .iter()
        .zip(&phi)
        .map(|(&(q, _), &p)| q * p)
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_salt_matches_kirkwood_inside() {
        // With κ → 0 and r_stern = r_sphere this should agree with the
        // existing κ=0 inside-charge Kirkwood formula for a pair of
        // z-axis charges. Compare two self + cross terms directly.
        use crate::analytical::kirkwood_inside::reaction_field_potential_unit_source;

        let eps_in = 2.0;
        let eps_out = 80.0;
        let r = 10.0;
        let z1 = 4.0;
        let z2 = 6.0;
        let q = 1.0;

        let u_salt = solvation_energy_z_axis(&[(q, z1), (q, z2)], eps_in, eps_out, r, r, 0.0, 40);

        // Direct two-charge reaction-field energy from the κ=0 formula.
        let phi_11 = reaction_field_potential_unit_source(
            [0.0, 0.0, z1],
            [0.0, 0.0, z1],
            r,
            eps_in,
            eps_out,
            40,
        );
        let phi_22 = reaction_field_potential_unit_source(
            [0.0, 0.0, z2],
            [0.0, 0.0, z2],
            r,
            eps_in,
            eps_out,
            40,
        );
        let phi_12 = reaction_field_potential_unit_source(
            [0.0, 0.0, z1],
            [0.0, 0.0, z2],
            r,
            eps_in,
            eps_out,
            40,
        );
        let u_k0 = 0.5 * (q * q * phi_11 + q * q * phi_22 + 2.0 * q * q * phi_12);

        let rel = (u_salt - u_k0).abs() / u_k0.abs();
        assert!(rel < 1e-10, "κ=0 agreement: {u_salt} vs {u_k0} rel {rel:e}");
    }
}
