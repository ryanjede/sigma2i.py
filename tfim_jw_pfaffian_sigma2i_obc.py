#!/usr/bin/env python3
"""
σ²_I Jordan-Wigner Exact Solution
==================================================

Purpose
-------
Estimate the pseudocritical peak h*(n) of σ²_I for the open TFIM using
the Jordan-Wigner / BdG free-fermion solution.

This version:
- uses entropy in bits (base 2), to match sigma2i_core.py
- uses a restricted search window near the TFIM critical region
- uses a 3-point local quadratic peak refinement
- prints step changes Δh*
- saves results to CSV

Validation
----------
Verified against exact diagonalisation (sigma2i_core.py) to machine
precision (|delta sigma^2_I| < 5e-15) at n = 4, 6, 8, 10. Trusted
for n <= 65; Pfaffian accuracy degrades for pair separations m > 70.
"""

import csv
import time
from itertools import combinations

import numpy as np


LOG2 = np.log(2.0)


# ====================== BdG SOLVER ======================

def solve_bdg(n: int, h: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the open-chain TFIM BdG problem.

    Returns
    -------
    C : ndarray, shape (n, n)
        <c†_i c_j>
    F : ndarray, shape (n, n)
        <c_i c_j>
    """
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, n), dtype=float)

    np.fill_diagonal(A, 2.0 * h)

    for i in range(n - 1):
        A[i, i + 1] = -1.0
        A[i + 1, i] = -1.0
        B[i, i + 1] = -1.0
        B[i + 1, i] = 1.0

    H = np.block([[A, B], [-B, -A]])
    vals, vecs = np.linalg.eigh(H)

    pos = vals > 1e-10
    U = vecs[:n, pos]
    V = vecs[n:, pos]

    C = np.real(V @ V.T)   # <c†_i c_j>
    F = np.real(U @ V.T)   # <c_i c_j>

    return C, F


# ====================== MAJORANA G MATRIX ======================

def build_G(C: np.ndarray, F: np.ndarray, n: int) -> np.ndarray:
    """
    Build the antisymmetric Majorana correlation matrix G.

    Conventions:
      A_i = c†_i + c_i
      B_i = i(c†_i - c_i)

    Using:
      G[A_i, B_j] = i(δ_ij - 2(C_ij + F_ij))
      G[A_i, A_j] = 0 for i != j
      G[B_i, B_j] = 0 for i != j
    """
    G = np.zeros((2 * n, 2 * n), dtype=complex)

    for i in range(n):
        G[2 * i, 2 * i + 1] = 1j * (1.0 - 2.0 * C[i, i])
        G[2 * i + 1, 2 * i] = -G[2 * i, 2 * i + 1]

        for j in range(n):
            if i == j:
                continue
            val = -2.0j * (C[i, j] + F[i, j])
            G[2 * i, 2 * j + 1] = val
            G[2 * j + 1, 2 * i] = -val

    return G


# ====================== PFAFFIAN ======================

def pfaffian(A: np.ndarray) -> complex:
    """
    Pfaffian of an even-dimensional antisymmetric matrix via elimination.
    """
    n = A.shape[0]

    if n == 0:
        return 1.0 + 0j
    if n % 2 != 0:
        raise ValueError("Pfaffian requires an even-dimensional matrix.")

    A = A.copy().astype(complex)
    pf = 1.0 + 0j

    for k in range(0, n - 2, 2):
        pivot = k + 1 + int(np.argmax(np.abs(A[k, k + 1:])))
        if np.abs(A[k, pivot]) < 1e-14:
            return 0.0 + 0j

        if pivot != k + 1:
            A[:, [k + 1, pivot]] = A[:, [pivot, k + 1]]
            A[[k + 1, pivot], :] = A[[pivot, k + 1], :]
            pf *= -1

        p = A[k, k + 1]
        pf *= p

        v = A[k, k + 2:].copy()
        w = A[k + 1, k + 2:].copy()
        A[k + 2:, k + 2:] -= (np.outer(v, w) - np.outer(w, v)) / p

    pf *= A[n - 2, n - 1]
    return pf


# ====================== ENTROPY HELPERS ======================

def binary_entropy_bits(p: float) -> float:
    p = float(np.clip(p, 1e-14, 1.0 - 1e-14))
    q = 1.0 - p
    return (-p * np.log(p) - q * np.log(q)) / LOG2


def entropy_from_eigs_bits(evals: list[float]) -> float:
    s = 0.0
    for lam in evals:
        lam = float(lam)
        if lam > 1e-14:
            s -= lam * np.log(lam)
    return s / LOG2


# ====================== MI FOR ONE PAIR ======================

def mi_pair_ff(C: np.ndarray, F: np.ndarray, G: np.ndarray, i: int, j: int) -> float:
    """
    Pairwise mutual information I(i:j) from JW free-fermion correlators.
    """
    if not (0 <= i < j < C.shape[0]):
        raise ValueError("Require 0 <= i < j < n.")

    m = j - i
    ai = 1.0 - 2.0 * C[i, i]
    aj = 1.0 - 2.0 * C[j, j]

    # zz correlator
    tzz = ai * aj + 4.0 * F[i, j] ** 2 - 4.0 * C[i, j] ** 2

    # xx correlator
    idx_xx = list(range(2 * i + 1, 2 * j + 1))
    pf_xx = pfaffian(G[np.ix_(idx_xx, idx_xx)])
    txx = float(np.real(((-1j) ** m) * pf_xx))

    # yy correlator
    idx_yy = [2 * i] + list(range(2 * i + 2, 2 * j)) + [2 * j + 1]
    pf_yy = pfaffian(G[np.ix_(idx_yy, idx_yy)])
    tyy = float(np.real(-(((-1j) ** m) * pf_yy)))

    c1_sq = (ai + aj) ** 2 + (txx - tyy) ** 2
    c2_sq = (ai - aj) ** 2 + (txx + tyy) ** 2
    c1 = float(np.sqrt(max(c1_sq, 0.0)))
    c2 = float(np.sqrt(max(c2_sq, 0.0)))

    evals = [
        (1.0 + tzz + c1) / 4.0,
        (1.0 + tzz - c1) / 4.0,
        (1.0 - tzz + c2) / 4.0,
        (1.0 - tzz - c2) / 4.0,
    ]

    evals = [max(0.0, min(1.0, ev)) for ev in evals]

    Si = binary_entropy_bits((1.0 + ai) / 2.0)
    Sj = binary_entropy_bits((1.0 + aj) / 2.0)
    Sij = entropy_from_eigs_bits(evals)

    mi = Si + Sj - Sij
    return float(max(0.0, mi))


# ====================== σ²_I FOR ONE (n, h) ======================

def sigma2_ff(n: int, h: float, return_mean: bool = False) -> float | tuple[float, float]:
    C, F = solve_bdg(n, h)
    G = build_G(C, F, n)

    mis = np.array(
        [mi_pair_ff(C, F, G, i, j) for i, j in combinations(range(n), 2)],
        dtype=float
    )

    sigma2 = float(np.var(mis))
    if return_mean:
        return sigma2, float(np.mean(mis))
    return sigma2


# ====================== PEAK REFINEMENT ======================

def refine_peak_3pt(x: np.ndarray, y: np.ndarray, k: int) -> tuple[float, str]:
    if not (0 < k < len(x) - 1):
        return float(x[k]), "grid"

    xx = x[k - 1:k + 2]
    yy = y[k - 1:k + 2]

    try:
        a, b, c = np.polyfit(xx, yy, 2)
        if a >= 0:
            return float(x[k]), "grid"

        h_star = -b / (2.0 * a)
        if xx[0] <= h_star <= xx[-1]:
            return float(h_star), "quadratic_3pt"
        return float(x[k]), "grid"

    except Exception:
        return float(x[k]), "grid"


# ====================== SMART PEAK FINDER ======================

def find_peak(
    n: int,
    coarse_min: float = 0.75,
    coarse_max: float = 1.10,
    coarse_steps: int = 40,
    fine_half_width: float = 0.05,
    fine_steps: int = 31,
    debug_dump_coarse: bool = False,
) -> dict:
    """
    Two-stage peak finder restricted to the physical TFIM critical region.
    """
    h_coarse = np.linspace(coarse_min, coarse_max, coarse_steps)
    s2_coarse = np.array([sigma2_ff(n, h) for h in h_coarse], dtype=float)

    k_coarse = int(np.argmax(s2_coarse))
    h_coarse_max = float(h_coarse[k_coarse])

    fine_lo = max(coarse_min, h_coarse_max - fine_half_width)
    fine_hi = min(coarse_max, h_coarse_max + fine_half_width)

    h_fine = np.linspace(fine_lo, fine_hi, fine_steps)
    s2_fine = np.array([sigma2_ff(n, h) for h in h_fine], dtype=float)

    k_fine = int(np.argmax(s2_fine))
    h_grid = float(h_fine[k_fine])
    h_refined, method = refine_peak_3pt(h_fine, s2_fine, k_fine)

    if debug_dump_coarse:
        print(f"\n[DEBUG] coarse scan for n={n}")
        for h, s in zip(h_coarse, s2_coarse):
            print(f"{h:.6f} {s:.12e}")

    return {
        "n": n,
        "h_star_refined": h_refined,
        "h_star_grid": h_grid,
        "method": method,
        "h_coarse_max": h_coarse_max,
        "coarse_h": h_coarse,
        "coarse_s2": s2_coarse,
        "fine_h": h_fine,
        "fine_s2": s2_fine,
        "fine_peak_sigma2": float(s2_fine[k_fine]),
        "fine_peak_idx": k_fine,
    }


# ====================== DRIVER ======================

def run_list(
    n_list: list[int],
    coarse_min: float = 0.75,
    coarse_max: float = 1.10,
    coarse_steps: int = 40,
    fine_half_width: float = 0.05,
    fine_steps: int = 31,
    csv_path: str = "sigma2i_jw_peaks.csv",
    debug_dump_n: set[int] | None = None,
) -> list[dict]:
    if debug_dump_n is None:
        debug_dump_n = set()

    print("Starting Jordan-Wigner σ²_I peak finder")
    print("=" * 88)
    print(
    f"{'n':>5}  {'h*_coarse':>12}  {'h*_grid':>12}  {'h*_refined':>12}  "
    f"{'delta':>12}  {'delta*n':>12}  {'delta*sqrt(n)':>14}  {'method':>14}  {'sec':>10}")
    print("-" * 88)

    rows: list[dict] = []

    for n in n_list:
        t0 = time.time()

        out = find_peak(
            n=n,
            coarse_min=coarse_min,
            coarse_max=coarse_max,
            coarse_steps=coarse_steps,
            fine_half_width=fine_half_width,
            fine_steps=fine_steps,
            debug_dump_coarse=(n in debug_dump_n),
        )

        if n in debug_dump_n:
            print(f"\n[DEBUG] fine scan for n={n}")
            for h, s in zip(out["fine_h"], out["fine_s2"]):
                print(f"{h:.6f}  {s:.12e}")
            print()

        elapsed = time.time() - t0

        row = {
            "n": n,
            "h_star_coarse": out["h_coarse_max"],
            "h_star_grid": out["h_star_grid"],
            "h_star_refined": out["h_star_refined"],
            "delta": 1.0 - out["h_star_refined"],
            "delta_n": (1.0 - out["h_star_refined"]) * n,
            "delta_sqrt_n": (1.0 - out["h_star_refined"]) * np.sqrt(n),
            "method": out["method"],
            "fine_peak_sigma2": out["fine_peak_sigma2"],
            "seconds": elapsed,
        }
        rows.append(row)

        print(
            f"{n:5d}  "
            f"{row['h_star_coarse']:12.6f}  "
            f"{row['h_star_grid']:12.6f}  "
            f"{row['h_star_refined']:12.6f}  "
            f"{row['delta']:12.6f}  "
            f"{row['delta_n']:12.6f}  "
            f"{row['delta_sqrt_n']:14.6f}  "
            f"{row['method']:>14}  "
            f"{elapsed:10.1f}"
        )

        if len(rows) >= 2:
            prev = float(rows[-2]["h_star_refined"])
            curr = float(rows[-1]["h_star_refined"])
            print(f"         Δh* = {curr - prev:+.6f}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            "h_star_coarse",
            "h_star_grid",
            "h_star_refined",
            "delta",
            "delta_n",
            "delta_sqrt_n",
            "method",
            "fine_peak_sigma2",
            "seconds",
        ])
        for row in rows:
            writer.writerow([
                row["n"],
                f"{row['h_star_coarse']:.12f}",
                f"{row['h_star_grid']:.12f}",
                f"{row['h_star_refined']:.12f}",
                f"{row['delta']:.12f}",
                f"{row['delta_n']:.12f}",
                f"{row['delta_sqrt_n']:.12f}",
                row["method"],
                f"{row['fine_peak_sigma2']:.16e}",
                f"{row['seconds']:.3f}",
            ])

    print("\nDone.")
    print(f"Saved: {csv_path}")

    return rows


# ====================== MAIN ======================

def fit_scaling_fixed_nu(rows, n_min=30):
    from scipy.optimize import curve_fit
    filtered = [r for r in rows if r["n"] >= n_min]

    ns = np.array([r["n"] for r in filtered], dtype=float)
    hs = np.array([r["h_star_refined"] for r in filtered], dtype=float)

    def scaling_fixed(n, hc, A):
        return hc + A / n

    popt, pcov = curve_fit(
        scaling_fixed,
        ns,
        hs,
        p0=[1.0, -1.0],
        bounds=([0.9, -10.0], [1.1, 0.0]),
        maxfev=50000,
    )

    hc, A = popt
    perr = np.sqrt(np.diag(pcov))

    print("\nScaling fit with 1/nu fixed to 1:")
    print(f"  points used: n >= {n_min}")
    print(f"  h_c = {hc:.6f} ± {perr[0]:.6f}")
    print(f"  A   = {A:.6f} ± {perr[1]:.6f}")

    return popt, pcov


if __name__ == "__main__":
    n_list = [8, 20, 40, 50, 60]

    results = run_list(
        n_list=n_list,
        coarse_min=0.940,
        coarse_max=1.0,
        coarse_steps=31,
        fine_half_width=0.010,
        fine_steps=61,
        csv_path="sigma2i_jw_peaks.csv",
        debug_dump_n=set(),   # example: {24, 26}
    )

    if len(results) >= 2:
        fit_scaling_fixed_nu(results, n_min=30)
