//! Modified spherical Bessel functions `i_n` and `k_n`, needed for the
//! salted Kirkwood reference.
//!
//! Convention: `i_0(x) = sinh(x)/x`, `k_0(x) = exp(−x)/x`; both satisfy the
//! modified spherical Bessel equation `x² y'' + 2x y' − [x² + n(n+1)] y = 0`.
//! Derivatives computed from the standard recurrence
//!   `d/dx i_n(x) = i_{n−1}(x) − (n+1)/x · i_n(x)`
//!   `d/dx k_n(x) = −k_{n−1}(x) − (n+1)/x · k_n(x)`
//! with `i_{−1}(x) = cosh(x)/x` and `k_{−1}(x) = exp(−x)/x` (= `k_0(x)` since
//! `k_{−n−1} = k_n`).

/// Modified spherical Bessel function of the first kind, `i_n(x)`, for
/// `n` from 0 to `n_max`. Returned vector has length `n_max + 1`.
///
/// Uses Miller's downward-recurrence algorithm: start from a large index
/// `N ≫ n_max` with arbitrary seed values, recurse downward via
///   `i_{n−1}(x) = i_{n+1}(x) + (2n+1)/x · i_n(x)`
/// and renormalise using the exact `i_0(x) = sinh(x)/x`. Downward
/// recurrence is unconditionally stable for `i_n`; the upward variant
/// suffers catastrophic cancellation once `(2n+1)/x · i_n ≫ i_{n−1}`.
pub fn i_series(x: f64, n_max: usize) -> Vec<f64> {
    if x.abs() < 1e-12 {
        // Near-zero limit: i_n(0) = δ_{n,0}. Exact.
        let mut out = vec![0.0; n_max + 1];
        out[0] = 1.0;
        return out;
    }

    // why: overshoot N so that the arbitrary seed decays to negligible by
    // the time we reach n_max. At small x the downward multipliers
    // (2n+1)/x grow values by many decades per step, so we rescale
    // whenever the running value exceeds RESCALE_THRESHOLD to prevent
    // overflow; the final normalization by the exact i_0 = sinh(x)/x
    // undoes all rescaling in one step.
    const RESCALE_THRESHOLD: f64 = 1e100;
    let n_extra = 30;
    let big_n = n_max + n_extra;
    let mut vals = vec![0.0; big_n + 2];
    vals[big_n] = 1.0;
    for n in (1..=big_n).rev() {
        let nf = n as f64;
        vals[n - 1] = vals[n + 1] + (2.0 * nf + 1.0) / x * vals[n];
        if vals[n - 1].abs() > RESCALE_THRESHOLD {
            let inv = 1.0 / vals[n - 1];
            for v in vals.iter_mut().take(big_n + 2).skip(n - 1) {
                *v *= inv;
            }
        }
    }
    let scale = x.sinh() / x / vals[0];
    vals.iter_mut().take(n_max + 1).for_each(|v| *v *= scale);
    vals.truncate(n_max + 1);
    vals
}

/// Modified spherical Bessel function of the second kind, `k_n(x)`, for
/// `n` from 0 to `n_max`. Returned vector has length `n_max + 1`.
///
/// Uses upward recurrence
///   `k_{n+1}(x) = k_{n−1}(x) + (2n+1)/x · k_n(x)`
/// seeded with `k_0 = exp(−x)/x`, `k_1 = exp(−x)(1 + 1/x)/x`. This direction
/// is unconditionally stable for `k_n`.
pub fn k_series(x: f64, n_max: usize) -> Vec<f64> {
    assert!(x > 0.0, "k_n diverges at x = 0");
    let mut out = Vec::with_capacity(n_max + 1);
    let e = (-x).exp();
    let k0 = e / x;
    let k1 = e * (1.0 + 1.0 / x) / x;
    out.push(k0);
    if n_max == 0 {
        return out;
    }
    out.push(k1);
    for n in 1..n_max {
        let nf = n as f64;
        let next = out[n - 1] + (2.0 * nf + 1.0) / x * out[n];
        out.push(next);
    }
    out
}

/// Derivative `d/dx i_n(x)` using `i_n' = i_{n−1} − (n+1)/x · i_n`.
/// For `n = 0`, `i_0'(x) = cosh(x)/x − sinh(x)/x²`.
pub fn i_deriv(n: usize, x: f64, i_vals: &[f64]) -> f64 {
    if n == 0 {
        if x.abs() < 1e-6 {
            // i_0(x) = sinh(x)/x ≈ 1 + x²/6; derivative ≈ x/3.
            return x / 3.0;
        }
        return x.cosh() / x - x.sinh() / (x * x);
    }
    // i_{n−1}: n ≥ 1 ⇒ index valid.
    let i_nm1 = i_vals[n - 1];
    let i_n = i_vals[n];
    let nf = n as f64;
    i_nm1 - (nf + 1.0) / x * i_n
}

/// Derivative `d/dx k_n(x) = −k_{n−1} − (n+1)/x · k_n`.
/// For `n = 0`, `k_0'(x) = −exp(−x)/x − exp(−x)/x² = −k_0(x) − k_0(x)/x`.
pub fn k_deriv(n: usize, x: f64, k_vals: &[f64]) -> f64 {
    if n == 0 {
        let k0 = k_vals[0];
        return -k0 - k0 / x;
    }
    let k_nm1 = k_vals[n - 1];
    let k_n = k_vals[n];
    let nf = n as f64;
    -k_nm1 - (nf + 1.0) / x * k_n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i0_matches_sinh_over_x() {
        for &x in &[0.1, 1.0, 3.0, 10.0] {
            let is = i_series(x, 0);
            assert!((is[0] - x.sinh() / x).abs() < 1e-14);
        }
    }

    #[test]
    fn k0_matches_exp_over_x() {
        for &x in &[0.1, 1.0, 3.0, 10.0] {
            let ks = k_series(x, 0);
            assert!((ks[0] - (-x).exp() / x).abs() < 1e-14);
        }
    }

    #[test]
    fn k1_matches_closed_form() {
        // k_1(x) = e^{-x}(1 + 1/x)/x.
        for &x in &[0.1, 1.0, 3.0] {
            let ks = k_series(x, 1);
            let expected = (-x).exp() * (1.0 + 1.0 / x) / x;
            assert!((ks[1] - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn i_k_satisfy_modified_spherical_bessel_equation() {
        // x² y'' + 2x y' − [x² + n(n+1)] y = 0 for y = i_n or k_n.
        // Verify numerically at a few (x, n) by finite differencing.
        //
        // why: h ~ 1e-4 balances truncation O(h²) and roundoff O(ε/h²);
        // expected residual ~1e-5 in double precision.
        let x = 2.0;
        let h = 1e-4;
        for n in [0usize, 1, 2, 5] {
            for f in [i_series, k_series] {
                let y_m = f(x - h, n)[n];
                let y_0 = f(x, n)[n];
                let y_p = f(x + h, n)[n];
                let yp = (y_p - y_m) / (2.0 * h);
                let ypp = (y_p - 2.0 * y_0 + y_m) / (h * h);
                let lhs = x * x * ypp + 2.0 * x * yp - (x * x + (n * (n + 1)) as f64) * y_0;
                assert!(
                    lhs.abs() / y_0.abs().max(1.0) < 1e-4,
                    "n={n}, residual={lhs:e}"
                );
            }
        }
    }

    #[test]
    fn derivatives_match_finite_difference() {
        let x = 2.0;
        let n_max = 5;
        let i_vals = i_series(x, n_max);
        let k_vals = k_series(x, n_max);
        let h = 1e-6;
        for n in 0..=n_max {
            let i_p = i_series(x + h, n)[n];
            let i_m = i_series(x - h, n)[n];
            let fd_i = (i_p - i_m) / (2.0 * h);
            let an_i = i_deriv(n, x, &i_vals);
            assert!(
                (fd_i - an_i).abs() < 1e-6,
                "i_{n}': FD={fd_i:e} vs an={an_i:e}"
            );

            let k_p = k_series(x + h, n)[n];
            let k_m = k_series(x - h, n)[n];
            let fd_k = (k_p - k_m) / (2.0 * h);
            let an_k = k_deriv(n, x, &k_vals);
            assert!(
                (fd_k - an_k).abs() < 1e-6,
                "k_{n}': FD={fd_k:e} vs an={an_k:e}"
            );
        }
    }
}
