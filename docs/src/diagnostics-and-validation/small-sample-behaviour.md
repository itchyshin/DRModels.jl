# Small-sample behaviour of phylogenetic covariance estimates

This page documents **measured** small-sample behaviour of the among-axis phylogenetic covariance
`Σ_a` in bivariate location–scale fits. It exists so that a number you get from a small tree is not
mistaken for a number you can act on.

Nothing here is a bug report. These are properties of first-order Laplace/REML estimation of variance
components, they shrink as sample size grows, and they are shared with comparable implementations.
They are documented rather than patched because removing them requires a higher-order correction, not
a guard.

## What was measured

1000 replicates, `N = 128`, true `Σ_a` diagonal in 0.16–0.25, reported as log-Cholesky entries `Lij`.

| entry | axis | Cholesky depth | bias |
|---|---|---|---|
| `L11` | μ₁×μ₁ | 0 | −0.093 |
| `L22` | μ₂×μ₂ | 1 | −0.282 |
| `L33` | σ₁×σ₁ | 2 | −0.899 |
| `L44` | σ₂×σ₂ | 3 | −1.852 |

All four are biased **downward** — variance components are under-estimated on small trees.

### It is small-sample, not structural

Bias × N is near-constant across an N-ladder (233 / 247 / 193 for `L44` at N = 128 / 256 / 512),
i.e. bias ≈ O(1/N). It shrinks with more tips; it does not run away.

### The pattern follows Cholesky depth, not "mean vs scale"

It is tempting to read the table as "mean axes are fine, scale axes collapse". That is not what the
numbers say. `L22` is a **mean** axis and already jumps 3×, comparable to the mean→scale step. Each
diagonal entry is `Var(axis k | axes 1..k−1)`, so it absorbs the compounding error of every axis
conditioned before it. Depth explains more of the escalation than the mean/scale distinction.

### Three contributors, measured separately

- **Cholesky-depth compounding** — affects every axis, mean and scale alike. Dominant.
- **The exp() link on scale axes** — real and correctly signed. The likelihood is exactly quadratic in
  a mean-axis latent (Laplace is *exact* there, to machine precision) but not in a scale-axis latent.
  On its own this accounts for only ~0.02 in log-SD at N = 128 — an order of magnitude too small to
  explain the observed −1.85.
- **Phylogenetic tip correlation** — an amplifier, roughly 2× for scale axes at `ntip = 16`. On a star
  topology the scale-axis bias roughly halves. Notably the `L44`/`L33` ratio (≈1.85) is *unchanged* by
  tree correlation, confirming that part is structural to the fixed axis ordering.

## REML helps substantially and is the right default here

Same-seed comparison at `ntip = 16`:

| entry | ML | REML | reduction |
|---|---|---|---|
| `L33` | −3.33 | **−1.42** | −57 % |
| `L44` | −6.02 | **−2.48** | −59 % |

The bordered Schur correction does include the mean↔scale cross-blocks and removes most of ML's
downward bias. What remains originates in the Laplace approximation of the random-effect integral
itself, which REML does not target.

If you are estimating `Σ_a` on a small tree, prefer `method = :REML`.

## Standard errors on the off-diagonals are not informative at small N

This matters more in practice than the bias itself, and it is easy to miss.

Correlation between `|error|` and the reported SE, over the same replicates:

| entry | corr(\|error\|, SE) | reading |
|---|---|---|
| `L11` (diagonal, mean)  | **+0.702** | SE tracks error — usable |
| `L44` (diagonal, scale) | **+0.786** | SE tracks error — usable |
| `L43` (off-diagonal)    | **+0.032** | SE carries almost no information about error |

For `L44` specifically, the mean reported SE (3.498) is far larger than the empirical SD of the
estimates (1.243). Coverage is therefore ~100 % — which sounds reassuring and is not: an interval wide
enough to always contain the truth conceals a bias of −1.85 (about 1.5 sampling SDs) rather than
revealing it.

**Do not read a wide interval on a small tree as evidence the estimate is fine.**

## Practical guidance

- **Diagonal entries** (variances) at small `ntip` are under-estimated; treat them as lower bounds.
- **Off-diagonal entries** (covariances between axes) at small `ntip`: treat the point estimate as
  weak and the SE as roughly uninformative. Do not build an inference on a single small-tree fit.
- **Prefer `method = :REML`** for variance components; it removes ~55–60 % of ML's bias here.
- **Scale axes need more tips than mean axes** for the same reliability, both because of the exp()
  link and because they sit deeper in the Cholesky ordering.
- **Reordering the axes does not fix this** — it moves which entry absorbs the worst compounding.

## Provenance

Measured 2026-08-25 on a prespecified N-ladder and coverage study. The
reproducibility record retains the full calibration and bias investigation.

The three-factor decomposition is not fully separated — the isolated exp()-link mechanism is 10–50×
too small alone, so the explanation is a compound. The star-vs-balanced comparison is 15 replicates
and is suggestive rather than production-grade.
