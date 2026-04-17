#!/usr/bin/env python3
import numpy as np
import scipy.linalg as la
from itertools import combinations
import time
import argparse

"""
Usage
-----
Example:

    python3 tfim_jw_determinant_sigma2i_obc.py -n 500 --h-min 0.90 --h-max 1.05 --h-step 0.001

Optional flags:

    --print-each-point
    --log-base {2,e}
    --max-local-peaks 5
    --min-sep-h 0.01
    --show-top-raw-points
"""

# ============================================================
# CORE TFIM / JW / SPIN-MI ROUTINES
# ============================================================

def solve_tfim_toeplitz_elements(n: int, h: float) -> np.ndarray:
    """
    Build the open-boundary TFIM Bogoliubov matrices and return
    the G matrix used for spin correlator reconstruction.
    """
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, n), dtype=float)

    np.fill_diagonal(A, 2.0 * h)

    for i in range(n - 1):
        A[i, i + 1] = -1.0
        A[i + 1, i] = -1.0
        B[i, i + 1] = -1.0
        B[i + 1, i] = 1.0

    mat_m = (A - B) @ (A + B)
    vals, vec_phi = la.eigh(mat_m)
    vals = np.sqrt(np.maximum(vals, 1e-18))

    vec_psi = (A + B) @ vec_phi / vals
    G = -(vec_phi @ vec_psi.T)
    return G


def binary_entropy_from_magnetisation(m: float, log_base: str = "2") -> float:
    p = np.clip((1.0 + m) / 2.0, 1e-15, 1.0 - 1e-15)

    if log_base == "2":
        log_fn = np.log2
    else:
        log_fn = np.log

    return -p * log_fn(p) - (1.0 - p) * log_fn(1.0 - p)


def two_site_entropy_from_evals(evals: np.ndarray, log_base: str = "2") -> float:
    evals = np.clip(np.asarray(evals, dtype=float), 1e-15, 1.0)

    if log_base == "2":
        log_fn = np.log2
    else:
        log_fn = np.log

    return float(-np.sum(evals * log_fn(evals)))


def get_spin_mi(i: int, j: int, G: np.ndarray, log_base: str = "2") -> float:
    """
    Compute spin mutual information I(i:j).
    """
    if i > j:
        i, j = j, i
    if i == j:
        return 0.0

    mz_i = -G[i, i]
    mz_j = -G[j, j]

    s_i = binary_entropy_from_magnetisation(mz_i, log_base=log_base)
    s_j = binary_entropy_from_magnetisation(mz_j, log_base=log_base)

    tzz = mz_i * mz_j - G[i, j] * G[j, i]

    txx = la.det(G[i:j, i + 1:j + 1])
    tyy = la.det(G[i + 1:j + 1, i:j])

    c1 = np.sqrt((mz_i + mz_j) ** 2 + (txx - tyy) ** 2)
    c2 = np.sqrt((mz_i - mz_j) ** 2 + (txx + tyy) ** 2)

    evals = np.array([
        (1.0 + tzz + c1) / 4.0,
        (1.0 + tzz - c1) / 4.0,
        (1.0 - tzz + c2) / 4.0,
        (1.0 - tzz - c2) / 4.0,
    ], dtype=float)

    s_ij = two_site_entropy_from_evals(evals, log_base=log_base)
    return max(0.0, float(s_i + s_j - s_ij))


def get_sigma2_i(n: int, h: float, log_base: str = "2") -> float:
    """
    σ²_I = variance of pairwise mutual informations over all i<j.
    """
    G = solve_tfim_toeplitz_elements(n, h)
    mis = [get_spin_mi(i, j, G, log_base=log_base) for i, j in combinations(range(n), 2)]
    return float(np.var(mis))


# ============================================================
# SCAN / PEAK UTILITIES
# ============================================================

def run_scan(n: int, h_values: np.ndarray, log_base: str = "2", print_each_point: bool = False):
    results = []

    for h in h_values:
        s2 = get_sigma2_i(n, float(h), log_base=log_base)
        results.append((float(h), s2))

        if print_each_point:
            print(f"h: {h:.4f} | Sigma2_I: {s2:.8e}")

    return results


def top_k_points(results, k=5):
    return sorted(results, key=lambda x: x[1], reverse=True)[:k]


def refine_peak_quadratic(results):
    h_vals = np.array([r[0] for r in results], dtype=float)
    s2_vals = np.array([r[1] for r in results], dtype=float)

    idx = int(np.argmax(s2_vals))

    if idx == 0 or idx == len(h_vals) - 1:
        return h_vals[idx], s2_vals[idx], False

    x = h_vals[idx - 1: idx + 2]
    y = s2_vals[idx - 1: idx + 2]

    a, b, c = np.polyfit(x, y, 2)

    if a >= 0:
        return h_vals[idx], s2_vals[idx], False

    h_star = -b / (2.0 * a)
    s2_star = a * h_star**2 + b * h_star + c
    return float(h_star), float(s2_star), True


def find_local_maxima(results, min_sep_h=0.01, max_peaks=5):
    """
    Find local maxima separated by at least min_sep_h.

    This avoids reporting adjacent points on the same broad crest
    as separate 'peaks'.
    """
    h_vals = np.array([r[0] for r in results], dtype=float)
    s2_vals = np.array([r[1] for r in results], dtype=float)

    candidates = []

    for i in range(1, len(s2_vals) - 1):
        if s2_vals[i] >= s2_vals[i - 1] and s2_vals[i] >= s2_vals[i + 1]:
            candidates.append((h_vals[i], s2_vals[i]))

    candidates.sort(key=lambda x: x[1], reverse=True)

    selected = []
    for h, s2 in candidates:
        if all(abs(h - h0) >= min_sep_h for h0, _ in selected):
            selected.append((h, s2))
        if len(selected) >= max_peaks:
            break

    return selected


def print_summary(
    n: int,
    h_values: np.ndarray,
    results,
    elapsed: float,
    max_local_peaks: int = 5,
    min_sep_h: float = 0.01,
    show_top_raw_points: bool = False,
    top_raw_k: int = 5,
):
    h_vals = np.array([r[0] for r in results], dtype=float)
    s2_vals = np.array([r[1] for r in results], dtype=float)

    idx = int(np.argmax(s2_vals))
    h_star = h_vals[idx]
    s2_star = s2_vals[idx]

    refined_h, refined_s2, used_fit = refine_peak_quadratic(results)
    local_peaks = find_local_maxima(results, min_sep_h=min_sep_h, max_peaks=max_local_peaks)

    print("-" * 50)
    print(f"AUDIT COMPLETE in {elapsed:.2f}s.")
    print(f"N = {n}")
    print(f"h range: {h_values[0]:.4f} to {h_values[-1]:.4f}")
    print(f"Grid step ≈ {h_values[1] - h_values[0]:.6f}" if len(h_values) > 1 else "Single-point scan")
    print(f"Grid peak h* = {h_star:.6f}")
    print(f"Grid Sigma2_I* = {s2_star:.8e}")

    if used_fit:
        print(f"Refined h* = {refined_h:.6f}")
        print(f"Refined Sigma2_I* = {refined_s2:.8e}")
    else:
        print("Refined peak = skipped (boundary maximum or unsuitable fit)")

    print("\nLocal maxima:")
    if local_peaks:
        for h, s2 in local_peaks:
            print(f"h: {h:.4f} | Sigma2_I: {s2:.8e}")
    else:
        print("No interior local maxima found in this scan window.")

    if show_top_raw_points:
        print("\nTop raw points:")
        for h, s2 in top_k_points(results, k=top_raw_k):
            print(f"h: {h:.4f} | Sigma2_I: {s2:.8e}")

def build_h_range(h_min: float, h_max: float, h_step: float) -> np.ndarray:
    if h_step <= 0:
        raise ValueError("h_step must be positive")
    if h_max < h_min:
        raise ValueError("h_max must be >= h_min")

    # Include endpoint
    n_steps = int(round((h_max - h_min) / h_step))
    return h_min + np.arange(n_steps + 1, dtype=float) * h_step


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Determinant-based Jordan-Wigner scan of sigma^2_I for the "
            "open-boundary transverse-field Ising model."
        )
    )

    parser.add_argument(
        "-n", "--n",
        type=int,
        nargs="+",
        default=[50],
        help="One or more system sizes, e.g. -n 4 5 6 8"
    )
    parser.add_argument(
        "--h-min",
        type=float,
        default=0.90,
        help="Minimum h value (default: 0.90)"
    )
    parser.add_argument(
        "--h-max",
        type=float,
        default=1.05,
        help="Maximum h value (default: 1.05)"
    )
    parser.add_argument(
        "--h-step",
        type=float,
        default=0.001,
        help="Step size in h (default: 0.001)"
    )
    parser.add_argument(
        "--log-base",
        choices=["2", "e"],
        default="2",
        help="Entropy log base: 2 or e (default: 2)"
    )
    parser.add_argument(
        "--print-each-point",
        action="store_true",
        help="Print sigma^2_I at every scan point"
    )
    parser.add_argument(
        "--max-local-peaks",
        type=int,
        default=5,
        help="Maximum number of local maxima to report (default: 5)"
    )
    parser.add_argument(
        "--min-sep-h",
        type=float,
        default=0.01,
        help="Minimum h separation between reported local peaks (default: 0.01)"
    )
    parser.add_argument(
        "--show-top-raw-points",
        action="store_true",
        help="Show top raw grid points in the summary"
    )
    parser.add_argument(
        "--top-raw-k",
        type=int,
        default=5,
        help="Number of top raw points to show if enabled (default: 5)"
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    h_range = build_h_range(args.h_min, args.h_max, args.h_step)

    for n in args.n:
        print(f"--- TFIM JW determinant scan (N={n}) ---")
        start = time.time()

        results = run_scan(
            n=n,
            h_values=h_range,
            log_base=args.log_base,
            print_each_point=args.print_each_point,
        )

        elapsed = time.time() - start

        print_summary(
            n=n,
            h_values=h_range,
            results=results,
            elapsed=elapsed,
            max_local_peaks=args.max_local_peaks,
            min_sep_h=args.min_sep_h,
            show_top_raw_points=args.show_top_raw_points,
            top_raw_k=args.top_raw_k,
        )

        print()


if __name__ == "__main__":
    main()
