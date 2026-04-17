#!/usr/bin/env python3
"""
reproduce_main_result.py
Reproduce the headline result from Ede (2026):
  The σ²_I peak occurs at h* < h_c = 1.0 for the TFIM with OBC.

This script:
  1. Scans h/J for n = 6, 8, 10, 12, 14
  2. Finds h* (σ²_I peak location) for each n
  3. Verifies h* < 1.0 in every case
  4. Prints a summary table

Expected output:
    n =  6:  h* = 0.943  ✓  (h* < h_c)
    n =  8:  h* = 0.925  ✓  (h* < h_c)
    n = 10:  h* = 0.920  ✓  (h* < h_c)
    n = 12:  h* = 0.920  ✓  (h* < h_c)
    n = 14:  h* = 0.920  ✓  (h* < h_c)

Runtime: ~2 min on a laptop (n=14 is the bottleneck).

Run from repo root:
    python3 reproduce_main_result.py
"""

import sys
import time
import numpy as np
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh

sys.path.insert(0, ".")

from sigma2i_core import from_state_vector


def tfim_ground_state(n, h):
    """Ground state of H = -Σ ZᵢZᵢ₊₁ - h Σ Xᵢ (OBC)."""
    sz = csr_matrix(np.array([[1, 0], [0, -1]], dtype=float))
    sx = csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
    I2 = eye(2, format="csr")

    dim = 2**n
    H = csr_matrix((dim, dim), dtype=float)

    for i in range(n - 1):
        op = eye(1, format="csr")
        for j in range(n):
            op = kron(op, sz if j in (i, i+1) else I2, format="csr")
        H -= op

    for i in range(n):
        op = eye(1, format="csr")
        for j in range(n):
            op = kron(op, sx if j == i else I2, format="csr")
        H -= h * op

    vals, vecs = eigsh(H, k=1, which="SA")
    return vecs[:, 0]


def find_peak(n, h_values):
    """Scan h values and return (h_peak, sigma2_peak)."""
    best_h, best_s2 = None, -1.0
    for h in h_values:
        psi = tfim_ground_state(n, h)
        r = from_state_vector(psi, n_qubits=n)
        if r.sigma2 > best_s2:
            best_h, best_s2 = h, r.sigma2
    return best_h, best_s2


def main():
    sizes = [6, 8, 10, 12, 14]
    h_coarse = np.linspace(0.7, 1.3, 31)

    print("Reproducing headline result: σ²_I peak at h* < h_c = 1.0")
    print("=" * 55)
    print()

    all_pass = True
    results = []

    for n in sizes:
        t0 = time.time()

        # Coarse scan
        h_peak, s2_peak = find_peak(n, h_coarse)

        # Fine scan around peak
        h_fine = np.linspace(h_peak - 0.05, h_peak + 0.05, 21)
        h_peak, s2_peak = find_peak(n, h_fine)

        elapsed = time.time() - t0
        passed = h_peak < 1.0
        if not passed:
            all_pass = False

        mark = "✓" if passed else "✗"
        results.append((n, h_peak, s2_peak, elapsed, passed))
        print(f"  n = {n:2d}:  h* = {h_peak:.3f}  σ²_I = {s2_peak:.5f}  "
              f"{mark}  ({elapsed:.1f}s)")

    print()
    print("─" * 55)
    if all_pass:
        print("ALL PASSED: h* < h_c = 1.0 for every system size.")
        print("The σ²_I peak is a pre-critical phenomenon, consistent")
        print("with Ede (2026) Figs. 1–2.")
    else:
        print("SOME FAILED: check results above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
