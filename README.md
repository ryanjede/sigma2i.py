# σ²_I: Structural Observable for Finite Quantum Systems

Open-source implementation accompanying:

> R. J. Ede, "σ²_I: A Second-Moment Functional of Pairwise Correlations as a Structural Observable in Finite Quantum Systems," 
> submitted to Physical Review Letters (2026).

σ²_I = Var(I(Aᵢ:Aⱼ)) — the variance of pairwise mutual information over all subsystem pairs — is a basis-invariant, partition-free observable that detects *where* correlations live in finite quantum systems, not merely *how much* correlation exists.

## Scripts

### `sigma2i_core.py`
Core routines: exact diagonalisation of the TFIM, 
pairwise MI computation, and σ²_I evaluation. 
Suitable for small systems (n ≤ 18) and as a reference 
implementation.

### `tfim_jw_determinant_sigma2i_obc.py`
Fast Jordan–Wigner / Bogoliubov free-fermion 
implementation using the tridiagonal eigenproblem. 
Includes:- Two-stage coarse→fine peak search

Practical for n ≤ 800+. Verified against exact 
diagonalisation to machine precision 
(|Δσ²_I| < 5 × 10⁻¹⁵) at n = 4, 6, 8, 10.

### `tfim_jw_pfaffian_sigma2i_obc.py`
Jordan–Wigner free-fermion implementation using 
Pfaffian reconstruction of spin correlators via 
Majorana correlation matrices.

## Quick start

### Determinant scan
```bash
# Edit N and h_range at bottom of script, then:
python3 tfim_jw_determinant_sigma2i_obc.py
```

### Pfaffian scan
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

## Requirements

- Python 3.9+
- NumPy
- SciPy
- Matplotlib (optional, for plots)

## Citation

```bibtex
@article{Ede2026sigma2i,
  author  = {Ede, Ryan J.},
  title   = {$\sigma^2\_I$: A Second-Moment Functional of Pairwise 
             Correlations as a Structural Observable in Finite 
             Quantum Systems},
  journal = {Physical Review Letters},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT
