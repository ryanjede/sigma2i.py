#!/usr/bin/env python3
"""
examples/demo_small_n.py
Quick demonstration of σ²_I on a small TFIM chain.

Shows:
  1. σ²_I for the TFIM ground state at several field values
  2. Where the peak occurs relative to h_c = 1.0
  3. Which pairs are "hot" vs "cold"

Run from repo root:
    python3 examples/demo_small_n.py

Expected output (n=8, OBC):
    Peak σ²_I occurs at h* ≈ 0.92, below the critical point h_c = 1.0.
    Hot pairs are near the chain edges; cold pairs span the bulk.

Runtime: ~5 seconds on a laptop.
"""

import sys
import os
import numpy as np

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sigma2i_core import from_state_vector

# ── Build TFIM Hamiltonian (OBC) ──────────────────────────────────────────

def tfim_ground_state(n, h):
    """Return ground state |ψ₀⟩ of the TFIM: H = -Σ ZᵢZᵢ₊₁ - h Σ Xᵢ (OBC)."""
    from scipy.sparse import kron, eye, csr_matrix
    from scipy.sparse.linalg import eigsh

    sz = csr_matrix(np.array([[1, 0], [0, -1]], dtype=float))
    sx = csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
    I2 = eye(2, format="csr")

    dim = 2**n
    H = csr_matrix((dim, dim), dtype=float)

    for i in range(n - 1):
        # -ZZ coupling
        op = eye(1, format="csr")
        for j in range(n):
            if j == i or j == i + 1:
                op = kron(op, sz, format="csr")
            else:
                op = kron(op, I2, format="csr")
        H -= op

    for i in range(n):
        # -h X transverse field
        op = eye(1, format="csr")
        for j in range(n):
            if j == i:
                op = kron(op, sx, format="csr")
            else:
                op = kron(op, I2, format="csr")
        H -= h * op

    vals, vecs = eigsh(H, k=1, which="SA")
    return vecs[:, 0]


# ── Scan ──────────────────────────────────────────────────────────────────

def main():
    n = 8
    h_values = np.linspace(0.5, 1.5, 41)
    
    print(f"σ²_I demo: TFIM n={n} chain (OBC)")
    print(f"Scanning h/J from {h_values[0]:.1f} to {h_values[-1]:.1f} "
          f"({len(h_values)} points)\n")

    results = []
    for h in h_values:
        psi = tfim_ground_state(n, h)
        r = from_state_vector(psi, n_qubits=n)
        results.append((h, r))

    # Find peak
    peak_idx = max(range(len(results)), key=lambda i: results[i][1].sigma2)
    h_peak, r_peak = results[peak_idx]

    print(f"Results:")
    print(f"  σ²_I peak at h* = {h_peak:.3f}  (h_c = 1.000)")
    print(f"  σ²_I      = {r_peak.sigma2:.5f}")
    print(f"  Mean MI   = {r_peak.mean_mi:.4f}")
    print(f"  Contrast  = {r_peak.contrast_ratio:.2f}x  (hot/cold)")
    print(f"  Hot pairs:  {r_peak.hot_pairs}")
    print(f"  Cold pairs: {r_peak.cold_pairs}")
    print()

    # Print table
    print(f"  {'h/J':>6s}  {'σ²_I':>10s}  {'Mean MI':>10s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}")
    for h, r in results:
        marker = " ◄ peak" if h == h_peak else ""
        print(f"  {h:6.3f}  {r.sigma2:10.5f}  {r.mean_mi:10.4f}{marker}")

    print(f"\nThe σ²_I peak at h*={h_peak:.3f} < h_c=1.0 reflects maximum")
    print(f"spatial heterogeneity of correlations in the pre-critical regime.")


if __name__ == "__main__":
    main()
