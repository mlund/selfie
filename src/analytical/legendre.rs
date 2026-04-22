//! Legendre polynomials `P_n(x)` via the standard two-term recurrence
//! `n·P_n = (2n−1)·x·P_{n−1} − (n−1)·P_{n−2}`, seeded with `P_0 = 1`,
//! `P_1 = x`. Stable for `|x| ≤ 1`.

/// Returns `[P_0(x), P_1(x), …, P_n_max(x)]`.
pub(super) fn legendre_series(x: f64, n_max: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_max + 1);
    out.push(1.0);
    if n_max == 0 {
        return out;
    }
    out.push(x);
    for n in 2..=n_max {
        let nf = n as f64;
        let p = ((2.0 * nf - 1.0) * x * out[n - 1] - (nf - 1.0) * out[n - 2]) / nf;
        out.push(p);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values_at_half() {
        // P_0(0.5) = 1, P_1(0.5) = 0.5, P_2(0.5) = −0.125, P_3(0.5) = −0.4375
        let p = legendre_series(0.5, 3);
        assert!((p[0] - 1.0).abs() < 1e-14);
        assert!((p[1] - 0.5).abs() < 1e-14);
        assert!((p[2] + 0.125).abs() < 1e-14);
        assert!((p[3] + 0.4375).abs() < 1e-14);
    }

    #[test]
    fn p_n_at_one_equals_one() {
        let p = legendre_series(1.0, 20);
        for (n, &v) in p.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-12, "P_{n}(1) = {v}, expected 1");
        }
    }
}
