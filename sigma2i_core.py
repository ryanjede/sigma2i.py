#!/usr/bin/env python3
"""
sigma2i_core.py

σ²_I: Variance of pairwise mutual information as a structural observable
for quantum systems.

Reference
---------
Ede, R. J. (2026). σ²_I: A Second-Moment Functional of Pairwise Correlations
as a Structural Observable in Finite Quantum Systems.

Usage
-----
    import sigma2i_core
    import numpy as np

    # From a density matrix (full system, shape 2^n x 2^n)
    rho = ...
    result = sigma2i_core.from_density_matrix(rho, n_qubits=8)
    print(result.sigma2, result.mean_mi, result.hot_pairs)

    # From a state vector (shape 2^n)
    psi = ...
    result = sigma2i_core.from_state_vector(psi, n_qubits=8)

    # From tomography data (list of 2-qubit reduced density matrices)
    # rdms[(i,j)] = 4x4 density matrix for qubits i<j
    result = sigma2i_core.from_rdms(rdms, n_qubits=8)

    # Scan over a Hamiltonian parameter (e.g. transverse field)
    h_values = np.linspace(0.3, 1.8, 50)
    scan = sigma2i_core.scan(hamiltonian_fn, h_values, n_qubits=8)
    scan.plot()

Requirements
------------
- numpy >= 1.20
- scipy >= 1.7
- matplotlib (optional, for plotting)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core entropy and MI functions
# ─────────────────────────────────────────────────────────────────────────────

def _von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    """
    Von Neumann entropy S(ρ) = -Tr(ρ log_base ρ).

    Parameters
    ----------
    rho : ndarray, shape (d, d)
        Density matrix.
    base : float
        Logarithm base (default 2 → bits).

    Returns
    -------
    float
        Entropy in bits (or nats if base=e).
    """
    eigvals = np.linalg.eigvalsh(np.asarray(rho, dtype=complex))
    eigvals = np.real_if_close(eigvals, tol=1000)
    eigvals = eigvals[eigvals > 1e-14]
    if eigvals.size == 0:
        return 0.0
    return float(-np.sum(eigvals * np.log(eigvals)) / np.log(base))


def _partial_trace(rho: np.ndarray, n: int, keep: List[int]) -> np.ndarray:
    """
    Partial trace of an n-qubit density matrix, keeping qubits in `keep`.

    Parameters
    ----------
    rho : ndarray, shape (2^n, 2^n)
        Full density matrix.
    n : int
        Total number of qubits.
    keep : list of int
        Indices of qubits to retain.

    Returns
    -------
    ndarray, shape (2^k, 2^k) where k = len(keep)
    """
    k = len(keep)
    trace_over = [q for q in range(n) if q not in keep]

    rho = np.asarray(rho, dtype=complex)
    rho_t = rho.reshape([2] * n + [2] * n)

    order = keep + trace_over
    rho_t = np.transpose(rho_t, order + [o + n for o in order])

    d_keep = 2 ** k
    d_trace = 2 ** (n - k)
    rho_t = rho_t.reshape(d_keep, d_trace, d_keep, d_trace)
    out = np.einsum("iaja->ij", rho_t)
    return np.real_if_close(out, tol=1000)


def _rdm_from_state(psi: np.ndarray, n: int, sites: List[int]) -> np.ndarray:
    """
    Reduced density matrix for given sites from a pure state vector.

    Parameters
    ----------
    psi : ndarray, shape (2^n,)
        Normalised state vector.
    n : int
        Number of qubits.
    sites : list of int
        Qubit indices to keep.

    Returns
    -------
    ndarray, shape (2^k, 2^k)
    """
    k = len(sites)
    rest = [q for q in range(n) if q not in sites]
    psi = np.asarray(psi, dtype=complex).reshape([2] * n)
    psi_t = np.transpose(psi, sites + rest).reshape(2 ** k, 2 ** (n - k))
    rho = psi_t @ psi_t.conj().T
    return np.real_if_close(rho, tol=1000)


def pairwise_mi(
    psi: Optional[np.ndarray] = None,
    rho: Optional[np.ndarray] = None,
    rdms: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    n: Optional[int] = None,
    base: float = 2.0,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Compute all C(n,2) pairwise mutual information values.

    Supply exactly one of: psi (state vector), rho (density matrix),
    or rdms (pre-computed 2-site reduced density matrices).

    Parameters
    ----------
    psi : ndarray, shape (2^n,), optional
        Pure state vector.
    rho : ndarray, shape (2^n, 2^n), optional
        Full density matrix (pure or mixed).
    rdms : dict, optional
        Pre-computed RDMs: rdms[(i,j)] = 4x4 array for qubits i<j.
    n : int, optional
        Number of qubits (required if using rho or rdms).
    base : float
        Logarithm base for entropy (default 2 → bits).

    Returns
    -------
    mis : ndarray, shape (C(n,2),)
        Mutual information for each pair.
    pairs : list of (int, int)
        Pair indices in same order as mis.
    """
    supplied = sum(x is not None for x in (psi, rho, rdms))
    if supplied != 1:
        raise ValueError("Supply exactly one of: psi, rho, or rdms")

    if psi is not None:
        psi = np.asarray(psi, dtype=complex).ravel()
        if n is None:
            n = int(round(np.log2(len(psi))))
        psi = psi / np.linalg.norm(psi)

        single_entropies = [_von_neumann_entropy(_rdm_from_state(psi, n, [i]), base) for i in range(n)]

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        mis = []
        for i, j in pairs:
            rho_ij = _rdm_from_state(psi, n, [i, j])
            s_ij = _von_neumann_entropy(rho_ij, base)
            mis.append(max(0.0, single_entropies[i] + single_entropies[j] - s_ij))
        return np.array(mis, dtype=float), pairs

    if rho is not None:
        rho = np.asarray(rho, dtype=complex)
        if n is None:
            n = int(round(np.log2(rho.shape[0])))

        single_entropies = [_von_neumann_entropy(_partial_trace(rho, n, [i]), base) for i in range(n)]

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        mis = []
        for i, j in pairs:
            rho_ij = _partial_trace(rho, n, [i, j])
            s_ij = _von_neumann_entropy(rho_ij, base)
            mis.append(max(0.0, single_entropies[i] + single_entropies[j] - s_ij))
        return np.array(mis, dtype=float), pairs

    if n is None:
        raise ValueError("n must be provided when using rdms")

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    single_entropies: Dict[int, float] = {}

    for (i, j), rho_ij in rdms.items():
        rho_ij = np.asarray(rho_ij, dtype=complex).reshape(4, 4)
        rho_tensor = rho_ij.reshape(2, 2, 2, 2)
        if i not in single_entropies:
            rho_i = np.trace(rho_tensor, axis1=1, axis2=3)
            single_entropies[i] = _von_neumann_entropy(rho_i, base)
        if j not in single_entropies:
            rho_j = np.trace(rho_tensor, axis1=0, axis2=2)
            single_entropies[j] = _von_neumann_entropy(rho_j, base)

    missing_sites = [site for site in range(n) if site not in single_entropies]
    if missing_sites:
        raise KeyError(f"Could not infer single-site RDMs for sites: {missing_sites}")

    mis = []
    for i, j in pairs:
        if (i, j) in rdms:
            rho_ij = np.asarray(rdms[(i, j)], dtype=complex).reshape(4, 4)
        elif (j, i) in rdms:
            rho_ij = np.asarray(rdms[(j, i)], dtype=complex).reshape(4, 4)
        else:
            raise KeyError(f"RDM for pair ({i}, {j}) not provided")
        s_ij = _von_neumann_entropy(rho_ij, base)
        mis.append(max(0.0, single_entropies[i] + single_entropies[j] - s_ij))
    return np.array(mis, dtype=float), pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Sigma2IResult:
    """
    Result of a σ²_I computation.

    Attributes
    ----------
    sigma2 : float
        σ²_I = Var(I(A_i:A_j)) over all C(n,2) pairs.
    mean_mi : float
        Mean pairwise MI.
    mis : ndarray
        All C(n,2) MI values.
    pairs : list of (int,int)
        Pair indices.
    n : int
        Number of qubits.
    hot_pairs : list of (int,int)
        Top-10% pairs by MI (the 'hot' spots).
    cold_pairs : list of (int,int)
        Bottom-10% pairs by MI (the 'cold' spots).
    contrast_ratio : float
        Mean MI of hot pairs / mean MI of cold pairs.
        High value → strong spatial heterogeneity.
    """

    sigma2: float
    mean_mi: float
    mis: np.ndarray
    pairs: List[Tuple[int, int]]
    n: int
    hot_pairs: List[Tuple[int, int]] = field(default_factory=list)
    cold_pairs: List[Tuple[int, int]] = field(default_factory=list)
    contrast_ratio: float = 0.0

    def __post_init__(self) -> None:
        if len(self.mis) > 0:
            threshold_hi = np.percentile(self.mis, 90)
            threshold_lo = np.percentile(self.mis, 10)
            self.hot_pairs = [p for p, m in zip(self.pairs, self.mis) if m >= threshold_hi]
            self.cold_pairs = [p for p, m in zip(self.pairs, self.mis) if m <= threshold_lo]
            hot_mean = float(np.mean([m for m in self.mis if m >= threshold_hi]))
            cold_mean = float(np.mean([m for m in self.mis if m <= threshold_lo]))
            self.contrast_ratio = hot_mean / cold_mean if cold_mean > 1e-10 else np.inf

    def heterogeneity_map(self) -> Dict[Tuple[int, int], float]:
        """Return {pair: MI} dict sorted descending by MI."""
        return dict(sorted(zip(self.pairs, self.mis), key=lambda x: -x[1]))

    def __repr__(self) -> str:
        return (
            f"Sigma2IResult(n={self.n}, σ²_I={self.sigma2:.5f}, "
            f"mean_MI={self.mean_mi:.4f}, contrast={self.contrast_ratio:.2f}x, "
            f"hot_pairs={len(self.hot_pairs)}, cold_pairs={len(self.cold_pairs)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main API functions
# ─────────────────────────────────────────────────────────────────────────────

def from_state_vector(
    psi: np.ndarray,
    n_qubits: Optional[int] = None,
    base: float = 2.0,
) -> Sigma2IResult:
    """
    Compute σ²_I from a pure state vector.

    Parameters
    ----------
    psi : array_like, shape (2^n,)
        Normalised state vector. Will be normalised if not already.
    n_qubits : int, optional
        Number of qubits. Inferred from len(psi) if not given.
    base : float
        Entropy base (default 2 → bits).

    Returns
    -------
    Sigma2IResult
    """
    psi = np.asarray(psi, dtype=complex).ravel()
    n = n_qubits or int(round(np.log2(len(psi))))
    mis, pairs = pairwise_mi(psi=psi, n=n, base=base)
    return Sigma2IResult(
        sigma2=float(np.var(mis)),
        mean_mi=float(np.mean(mis)),
        mis=mis,
        pairs=pairs,
        n=n,
    )


def from_density_matrix(
    rho: np.ndarray,
    n_qubits: Optional[int] = None,
    base: float = 2.0,
) -> Sigma2IResult:
    """
    Compute σ²_I from a (possibly mixed) density matrix.

    Parameters
    ----------
    rho : array_like, shape (2^n, 2^n)
        Density matrix.
    n_qubits : int, optional
        Number of qubits.
    base : float
        Entropy base.

    Returns
    -------
    Sigma2IResult
    """
    rho = np.asarray(rho, dtype=complex)
    n = n_qubits or int(round(np.log2(rho.shape[0])))
    mis, pairs = pairwise_mi(rho=rho, n=n, base=base)
    return Sigma2IResult(
        sigma2=float(np.var(mis)),
        mean_mi=float(np.mean(mis)),
        mis=mis,
        pairs=pairs,
        n=n,
    )


def from_rdms(
    rdms: Dict[Tuple[int, int], np.ndarray],
    n_qubits: int,
    base: float = 2.0,
) -> Sigma2IResult:
    """
    Compute σ²_I from pre-computed 2-site reduced density matrices.

    This is the most efficient path for experimental tomography data
    where you already have the 2-qubit RDMs.

    Parameters
    ----------
    rdms : dict
        {(i, j): rho_ij} where rho_ij is a 4×4 density matrix
        for qubits i and j (i < j).
    n_qubits : int
        Total number of qubits in the system.
    base : float
        Entropy base.

    Returns
    -------
    Sigma2IResult
    """
    mis, pairs = pairwise_mi(rdms=rdms, n=n_qubits, base=base)
    return Sigma2IResult(
        sigma2=float(np.var(mis)),
        mean_mi=float(np.mean(mis)),
        mis=mis,
        pairs=pairs,
        n=n_qubits,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parameter scan
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    """
    Result of a σ²_I parameter scan.

    Attributes
    ----------
    params : ndarray
        Parameter values scanned.
    sigma2 : ndarray
        σ²_I at each parameter value.
    mean_mi : ndarray
        Mean MI at each parameter value.
    sigma2_norm : ndarray
        σ²_I normalised to its maximum.
    mean_mi_norm : ndarray
        Mean MI normalised to its maximum.
    peak_param : float
        Parameter value where σ²_I is maximum.
    peak_idx : int
        Index of the peak.
    results : list of Sigma2IResult
        Full result at each parameter value.
    """

    params: np.ndarray
    sigma2: np.ndarray
    mean_mi: np.ndarray
    sigma2_norm: np.ndarray
    mean_mi_norm: np.ndarray
    peak_param: float
    peak_idx: int
    results: List[Sigma2IResult]

    def plot(self, ax=None, show_mean_mi: bool = True, title: str = "σ²_I Parameter Scan"):
        """
        Plot σ²_I (and optionally mean MI) vs parameter.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib required for plotting") from exc

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(self.params, self.sigma2_norm, "b-", lw=2.5, label="σ²_I (normalised)")
        if show_mean_mi:
            ax.plot(self.params, self.mean_mi_norm, "r--", lw=2, label="Mean MI (normalised)")
        ax.axvline(
            self.peak_param,
            color="blue",
            ls=":",
            lw=1.2,
            alpha=0.8,
            label=f"σ²_I peak = {self.peak_param:.3f}",
        )

        ax.set_xlabel("Parameter", fontsize=11)
        ax.set_ylabel("Normalised value", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.25)
        return ax

    def heterogeneity_map_at_peak(self) -> Dict[Tuple[int, int], float]:
        """Return the pair heterogeneity map at the σ²_I peak."""
        return self.results[self.peak_idx].heterogeneity_map()


def scan(
    ground_state_fn: Callable,
    params: np.ndarray,
    n_qubits: int,
    input_type: str = "state_vector",
    base: float = 2.0,
    verbose: bool = True,
) -> ScanResult:
    """
    Scan σ²_I over a range of parameter values.

    Parameters
    ----------
    ground_state_fn : callable
        Function that takes a parameter value and returns either:
        - a state vector (input_type='state_vector'), or
        - a density matrix (input_type='density_matrix'), or
        - a dict of RDMs (input_type='rdms').
    params : array_like
        Parameter values to scan over.
    n_qubits : int
        Number of qubits.
    input_type : str
        One of 'state_vector', 'density_matrix', 'rdms'.
    base : float
        Entropy base.
    verbose : bool
        Print progress.

    Returns
    -------
    ScanResult
    """
    params = np.asarray(params, dtype=float)
    results: List[Sigma2IResult] = []
    s2_vals: List[float] = []
    mean_vals: List[float] = []

    for k, p in enumerate(params):
        if verbose and k % 10 == 0:
            print(f"  Scanning {k + 1}/{len(params)}: param={p:.3f}")

        data = ground_state_fn(p)

        if input_type == "state_vector":
            result = from_state_vector(data, n_qubits=n_qubits, base=base)
        elif input_type == "density_matrix":
            result = from_density_matrix(data, n_qubits=n_qubits, base=base)
        elif input_type == "rdms":
            result = from_rdms(data, n_qubits=n_qubits, base=base)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        results.append(result)
        s2_vals.append(result.sigma2)
        mean_vals.append(result.mean_mi)

    s2 = np.array(s2_vals, dtype=float)
    mean_mi = np.array(mean_vals, dtype=float)
    peak_idx = int(np.argmax(s2))

    return ScanResult(
        params=params,
        sigma2=s2,
        mean_mi=mean_mi,
        sigma2_norm=s2 / s2.max() if s2.max() > 0 else s2,
        mean_mi_norm=mean_mi / mean_mi.max() if mean_mi.max() > 0 else mean_mi,
        peak_param=float(params[peak_idx]),
        peak_idx=peak_idx,
        results=results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: TFIM exact diagonalisation (bundled demo)
# ─────────────────────────────────────────────────────────────────────────────

def _tfim_ground_state(n: int, h: float, bc: str = "open") -> np.ndarray:
    """
    Ground state of TFIM H = -ZZ - h*X via sparse exact diagonalisation.
    Bundled so the package is self-contained for demos up to n~18.
    """
    try:
        from scipy.sparse import csr_matrix, eye as speye, kron as spkron
        from scipy.sparse.linalg import eigsh
    except ImportError as exc:
        raise ImportError("scipy required for built-in TFIM solver") from exc

    z = csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
    x = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    i2 = speye(2, format="csr")
    H = csr_matrix((2 ** n, 2 ** n), dtype=float)

    for site in range(n - 1):
        op = speye(1, format="csr")
        for j in range(n):
            op = spkron(op, z if j in (site, site + 1) else i2, format="csr")
        H -= op

    if bc == "periodic":
        op = speye(1, format="csr")
        for j in range(n):
            op = spkron(op, z if j in (0, n - 1) else i2, format="csr")
        H -= op

    for site in range(n):
        op = speye(1, format="csr")
        for j in range(n):
            op = spkron(op, x if j == site else i2, format="csr")
        H -= h * op

    _, vecs = eigsh(H, k=1, which="SA", tol=1e-12)
    return np.asarray(vecs[:, 0], dtype=complex)


def demo_tfim(
    n: int = 8,
    h_range: Tuple[float, float] = (0.3, 1.8),
    n_points: int = 50,
    bc: str = "open",
    plot: bool = True,
) -> ScanResult:
    """
    Demo: scan σ²_I over TFIM transverse field for n-qubit chain.
    """
    params = np.linspace(h_range[0], h_range[1], n_points)

    def gs(h: float) -> np.ndarray:
        return _tfim_ground_state(n, h, bc=bc)

    print(f"σ²_I demo: TFIM n={n} {bc} boundary conditions")
    print(f"Scanning h/J from {h_range[0]} to {h_range[1]}, {n_points} points")

    result = scan(gs, params, n_qubits=n, input_type="state_vector", verbose=False)

    print("\nResults:")
    print(f"  σ²_I peak:    h* = {result.peak_param:.3f}")
    print(f"  Mean MI peak: h  = {result.params[np.argmax(result.mean_mi)]:.3f}")
    print("  True h_c = 1.000  (TFIM)")
    print("\nAt σ²_I peak:")
    r = result.results[result.peak_idx]
    print(f"  σ²_I = {r.sigma2:.5f}")
    print(f"  Mean MI = {r.mean_mi:.4f}")
    print(f"  Contrast ratio (hot/cold) = {r.contrast_ratio:.2f}x")
    print(f"  Hottest pairs: {r.hot_pairs[:3]}")
    print(f"  Coldest pairs: {r.cold_pairs[:3]}")

    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            result.plot(ax=axes[0], title=f"σ²_I scan: TFIM n={n} {bc.upper()}")
            axes[0].axvline(1.0, color="k", ls="--", lw=1, label="h_c=1.0")
            axes[0].legend(fontsize=9)

            hmap = result.heterogeneity_map_at_peak()
            pairs_sorted = list(hmap.keys())
            mi_sorted = list(hmap.values())
            distances = [abs(j - i) for i, j in pairs_sorted]
            scatter = axes[1].scatter(
                distances,
                mi_sorted,
                c=mi_sorted,
                cmap="RdYlBu_r",
                s=60,
                alpha=0.8,
            )
            plt.colorbar(scatter, ax=axes[1], label="MI (bits)")
            axes[1].set_xlabel("Pair distance |i-j|", fontsize=11)
            axes[1].set_ylabel("Mutual information (bits)", fontsize=11)
            axes[1].set_title(f"Heterogeneity map at h*={result.peak_param:.3f}", fontsize=11)
            axes[1].grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig("sigma2i_demo.png", dpi=120, bbox_inches="tight")
            print("\nPlot saved to sigma2i_demo.png")
        except ImportError:
            print("matplotlib not available — skipping plot")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Module info
# ─────────────────────────────────────────────────────────────────────────────

__version__ = "1.0.0"
__author__ = "Ryan J. Ede"
__email__ = "ryanjede@gmail.com"
__license__ = "MIT"

__all__ = [
    "from_state_vector",
    "from_density_matrix",
    "from_rdms",
    "scan",
    "demo_tfim",
    "pairwise_mi",
    "Sigma2IResult",
    "ScanResult",
]


if __name__ == "__main__":
    print(f"sigma2i v{__version__}")
    print("Running TFIM demo (n=8)...")
    demo_tfim(n=8, h_range=(0.3, 2.0), n_points=150, plot=True)
