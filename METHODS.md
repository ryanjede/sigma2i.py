# Methods Overview

This repository contains multiple computational paths for evaluating **σ²_I**, the variance of pairwise mutual information in finite quantum systems, with a focus on the open-boundary transverse-field Ising model (TFIM).

The code is organized so that different methods serve different roles: small-system reference evaluation, moderate-size validation, and larger-size scanning.

---

## Observable

The central observable is

**σ²_I = Var(I(A_i : A_j))**

where the variance is taken over all subsystem pairs `i < j`.

The purpose of the observable is to detect **where correlations live** in a finite system, not merely how much total correlation is present.

---

## Method map

### 1. Exact diagonalisation (ED) path

**File:** `sigma2i_core.py`

This is the small-system reference implementation.

It provides:
- exact diagonalisation of the TFIM
- pairwise mutual information calculation
- σ²_I evaluation from:
  - state vectors
  - density matrices
  - precomputed two-site reduced density matrices

Primary role:
- ground-truth reference for small systems
- API / demonstration path
- correctness anchor for overlap checks

Practical scope:
- suitable for small systems
- in practice, typically `n <= 18` on standard hardware

---

### 2. Jordan-Wigner + Pfaffian path

**File:** `tfim_jw_pfaffian_sigma2i_obc.py`

This path uses the Jordan-Wigner free-fermion solution together with Pfaffian reconstruction of spin correlators via Majorana correlation matrices.

Primary role:
- moderate-size validation path
- overlap check against the determinant implementation
- early regime-identification path

Interpretive role:
- this was the first implementation used to reveal the finite-size regime structure of σ²_I in the OBC TFIM workflow

Validation status:
- used as a cross-check against the ED/reference path in the validated overlap regime
- strongest confidence is in the validated overlap range
- low-70s usage should be treated as exploratory unless independently revalidated

---

### 3. Jordan-Wigner + determinant path

**File:** `tfim_jw_determinant_sigma2i_obc.py`

This path uses the Jordan-Wigner / Bogoliubov free-fermion formulation together with determinant-based spin-correlator reconstruction.

Primary role:
- larger-system scan path
- finite-size peak tracking
- extended scaling and structure analysis

This is the main practical route for larger `n` in the OBC TFIM.

Depending on the current public version, the script may include:
- command-line scan configuration
- local-maximum reporting
- coarse/fine peak search or fixed-grid scan
- printed scan summaries

---

## Validated vs exploratory use

It is important to distinguish between:

### Validated range
A range where explicit overlap checks against another method were performed.

### Practical range
A range that is computationally feasible on available hardware.

### Exploratory range
A range that appears informative or numerically usable, but is not being presented as equally validated.

These are not the same thing.

For this repository:

- **ED path**  
  Serves as the small-system ground-truth reference path.

- **Pfaffian path**  
  Used as a validated overlap/cross-check method in its established range.  
  Use beyond that should be described as exploratory unless separately revalidated.

- **Determinant path**  
  Used for larger scans and finite-size studies.  
  Its role is practical scaling and scan generation, not replacement of the small-system reference path.

---

## Recommended use

### If you are new to the repository
Start with:
- `sigma2i_core.py`

### If you want a moderate-size validation path
Use:
- `tfim_jw_pfaffian_sigma2i_obc.py`

### If you want larger TFIM scans
Use:
- `tfim_jw_determinant_sigma2i_obc.py`

---

## Important caution

This repository contains research code, not a polished general-purpose software package.

Some scripts are intended primarily for:
- reference evaluation
- overlap checking
- numerical exploration
- scan generation for ongoing research

Outputs should therefore be interpreted in the context of:
- method role
- overlap validation
- numerical range
- scan window choice
- computational limits

---

## Repository philosophy

The public code is intended to make the research legible and inspectable.

It should be read as:
- a reference implementation
- a validation and scan toolkit
- a computational companion to the associated research work

It should not be assumed that every script in the repository serves the same role or carries the same evidentiary weight.
