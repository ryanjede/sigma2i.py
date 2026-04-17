# σ²_I: Structural Observable for Finite Quantum Systems

Reproducible Python implementation of a basis-invariant, partition-free
structural observable for finite quantum systems.

σ²_I = Var(I(Aᵢ:Aⱼ)) — the variance of pairwise mutual information
over all subsystem pairs — detects *where* correlations live in finite
quantum systems, not merely *how much* correlation exists.

Companion code for:

> R. J. Ede, “σ²_I: A Second-Moment Functional of Pairwise
> Correlations as a Structural Observable in Finite Quantum Systems,”
> submitted to Physical Review Letters (2026).

## Scripts

### `sigma2i_core.py` — Reference implementation (ED)

Core routines: exact diagonalisation of the transverse-field Ising
model (TFIM), pairwise MI computation, and σ²_I evaluation.
Suitable for small systems (n ≤ 18) and as ground-truth reference
for validating other methods.

### `tfim_jw_determinant_sigma2i_obc.py` — Scalable scan path

Fast Jordan–Wigner / Bogoliubov free-fermion implementation using
the tridiagonal eigenproblem. Includes:

- Two-stage coarse→fine peak search
- Variance decomposition (law of total variance) for mechanistic
  analysis of the finite-size crossover
- CSV and PNG output

Practical for n ≤ 800+.

### `tfim_jw_pfaffian_sigma2i_obc.py` — Pfaffian validation path

Jordan–Wigner free-fermion implementation using Pfaffian
reconstruction of spin correlators via Majorana correlation
matrices.

## Validation

Both JW implementations are verified against exact diagonalisation
to machine precision (|Δσ²_I| < 5 × 10⁻¹⁵) at n = 4, 6, 8, 10.

|Method     |Trusted range|Use case                       |
|-----------|-------------|-------------------------------|
|ED (core)  |n ≤ 18       |Ground truth, any Hamiltonian  |
|Pfaffian   |n ≤ 65       |Cross-check of determinant path|
|Determinant|n ≤ 800+     |Large-n scaling scans          |

## Quick start

### Install dependencies

```bash
pip install numpy scipy matplotlib
```

### Run a determinant scan (large n)

```bash
# Edit N and h_range at bottom of script, then:
python3 tfim_jw_determinant_sigma2i_obc.py
```

### Run a Pfaffian scan (moderate n)

```bash
# Edit N and h_range at bottom of script, then:
python3 tfim_jw_pfaffian_sigma2i_obc.py
```

### Exact diagonalisation (n ≤ 18)

```python
from sigma2i_core import from_state_vector, demo_tfim

# Quick demo: TFIM n=8 scan
result = demo_tfim(n=8)

# From any state vector
result = from_state_vector(psi, n_qubits=8)
print(result.sigma2, result.mean_mi, result.contrast_ratio)
```

### Reproduce a headline result

The σ²_I peak for the TFIM with open boundary conditions occurs at
h* ≈ 0.925 (n = 14), below the critical point h_c = 1.0. Run the
determinant script with n = 14 and h_range covering [0.85, 1.05]
to verify.

## Requirements

- Python 3.9+
- NumPy
- SciPy
- Matplotlib (optional, for plots)

## Citation

```bibtex
@article{Ede2026sigma2i,
  author  = {Ede, Ryan J.},
  title   = {$\sigma^2_I$: A Second-Moment Functional of Pairwise
             Correlations as a Structural Observable in Finite
             Quantum Systems},
  journal = {Physical Review Letters},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT
