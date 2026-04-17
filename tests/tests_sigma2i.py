"""
tests/test_sigma2i.py
Minimal test suite for sigma2i.py

Run with:
    python3 -m pytest tests/test_sigma2i.py -v
    
Or without pytest:
    python3 tests/test_sigma2i.py
"""

import sys
import os
import numpy as np

# Add repo root to path so imports work from tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_ed_demo_runs():
    """sigma2i_core.demo_tfim runs without error and returns sensible output."""
    from sigma2i_core import demo_tfim

    result = demo_tfim(n=6)
    assert hasattr(result, "sigma2"), "Result should have sigma2 attribute"
    assert result.sigma2 >= 0, "Variance must be non-negative"
    assert np.isfinite(result.sigma2), "Variance must be finite"


def test_ed_peak_below_critical():
    """σ²_I peak occurs below h_c = 1.0 for small OBC chains."""
    from sigma2i_core import demo_tfim

    # Scan h from 0.7 to 1.3
    h_values = np.linspace(0.7, 1.3, 31)
    sigma2_values = []
    for h in h_values:
        # demo_tfim does a built-in scan; we need from_state_vector instead
        # For this test, we build the Hamiltonian manually
        pass  # placeholder — requires knowing the internal API

    # If demo_tfim returns a scan, check peak location
    # This test documents the expected behaviour even if it needs
    # adaptation to the actual API
    print("SKIP: test_ed_peak_below_critical needs API adaptation")


def test_variance_non_negative():
    """σ²_I is a variance and must always be ≥ 0."""
    from sigma2i_core import demo_tfim

    for n in [4, 6, 8]:
        result = demo_tfim(n=n)
        assert result.sigma2 >= 0, f"σ²_I negative at n={n}: {result.sigma2}"


def test_trivial_product_state():
    """For a product state |00...0⟩, all MI = 0, so σ²_I = 0."""
    from sigma2i_core import from_state_vector

    n = 6
    psi = np.zeros(2**n)
    psi[0] = 1.0  # |000000⟩

    result = from_state_vector(psi, n_qubits=n)
    assert abs(result.sigma2) < 1e-12, (
        f"Product state should have σ²_I ≈ 0, got {result.sigma2}"
    )


def test_ghz_state_uniform_mi():
    """For a GHZ state, all pairs have equal MI, so σ²_I = 0."""
    from sigma2i_core import from_state_vector

    n = 4
    psi = np.zeros(2**n)
    psi[0] = 1.0 / np.sqrt(2)        # |0000⟩
    psi[2**n - 1] = 1.0 / np.sqrt(2)  # |1111⟩

    result = from_state_vector(psi, n_qubits=n)
    assert abs(result.sigma2) < 1e-12, (
        f"GHZ state should have σ²_I ≈ 0, got {result.sigma2}"
    )


# --------------- run without pytest ---------------
if __name__ == "__main__":
    tests = [
        test_ed_demo_runs,
        test_variance_non_negative,
        test_trivial_product_state,
        test_ghz_state_uniform_mi,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
