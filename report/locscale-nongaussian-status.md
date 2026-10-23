# Non-Gaussian location–scale engine (q=2) — status

*Status as of 2026-06-06. Tracks #202. All claims below are CI-verified on the
DRM.jl test suite (Julia 1.10 + 1; the remote dev environment has no local Julia,
so verification this round was CI-only).*

## What it is

A non-Gaussian **location–scale** model: per group there is a latent effect on
**both** the mean axis (η, log-mean) **and** the log-dispersion axis (ψ), tied by
a 2×2 group-level covariance **Λ**. Families: **NB2** (size `r = exp ψ`) and
**Gamma** (shape `α = exp ψ`). The grouping precision `Q` is pluggable:

- `Q = I_G` → i.i.d. / crossed groups;
- `Q =` root-conditioned tree precision → the **phylogenetic** location–scale model
  (`_locscale_phylo_setup(tree, labels)`).

The marginal is a sparse augmented-state Laplace approximation; the latent layout
is group-major `a = [aᵍ_μ, aᵍ_ψ, …]` with prior precision `P = kron(Q, Λ⁻¹)`.

## Verified pieces (src/locscale_*.jl)

| Piece | Function | Gate |
|---|---|---|
| Two-axis kernels (nll, grad, Hess in η,ψ) | `_ls_nll/_ls_grad/_ls_hess` | vs ForwardDiff |
| Augmented inner mode-finder (LM Newton) | `_ls_inner_mode` | grad→0 + vs ForwardDiff |
| Laplace marginal | `_ls_marginal_nll` | vs 2-D Gauss–Hermite |
| **Exact O(p) outer gradient** | `_ls_marginal_grad` | vs finite differences (NB2 & Gamma; i.i.d. & tree) |
| Gradient-driven fit (LBFGS + feasibility guard) | `_fit_locscale` | stationarity ‖∇‖→0 + variance-component recovery |
| **Wald inference** (observed information, SE, vcov) | `_ls_obs_information`/`_ls_vcov`/`_ls_se` | obs-info vs nll 2nd differences; PD vcov |
| Group-level summaries (sd_mu, sd_psi, ρ_a) | `_ls_components` | Λ-consistency |

### The exact gradient (the hard part)

`M(θ) = jn(â) + ½ logdet H − ½ logdet P`, `g(â)=0`, adjoint form:

    dM/dθₖ = ∂jn/∂θₖ + ½ tr(H⁻¹ ∂H/∂θₖ) − ½ tr(P⁻¹ ∂P/∂θₖ) − wᵀ ∂g/∂θₖ

with `v_j = ½ tr(H⁻¹ ∂H/∂a_j)`, `w = H⁻¹ v`. `H⁻¹` blocks come from the **Takahashi
selected inverse** (never a dense inverse); the kernel **third derivatives** come
from ForwardDiff of the analytic `_ls_hess` (no hand 3rd-deriv algebra). The λ
block uses `P = kron(Q, Λ⁻¹)`, giving `−½ tr(P⁻¹ ∂P/∂λₖ) = +½ G·tr(Λ⁻¹ ∂Λ/∂λₖ)`.
Full derivation: `docs/dev-log/2026-06-06-locscale-exact-gradient.md`.

## Using the engine today (internal API)

```julia
# i.i.d. groups, NB2
Q = sparse(1.0I, G, G)
fit = DRM._fit_locscale(Val(:nb2), y, Xμ, Xψ, gidx, G, Q; se = true)
fit.beta_mu          # mean fixed effects
fit.beta_psi         # log-dispersion fixed effects
fit.Lambda           # 2×2 group-level covariance
fit.components       # (sd_mu, sd_psi, cor_mu_psi)
fit.se, fit.vcov     # Wald SEs / covariance (se=true)

# phylogenetic location–scale
Q, gidx, G = DRM._locscale_phylo_setup(tree, labels)
fit = DRM._fit_locscale(Val(:gamma), y, Xμ, Xψ, gidx, G, Q; se = true)
```

## Not done — slice 3b: public `bf()`/`drm()` routing (needs decisions)

The engine is **not yet wired into `drm()`**. Wiring is mechanically tractable
(each family already has its own `drm()` method consuming `(y, Xμ, Xψ, gidx, G, Q)`),
but three user-facing decisions are open (tracked in #209):

1. **Family handle** — a new `LocationScale` switch, or reuse `NegBinomial2()`/`Gamma()`
   with a location–scale flag? (drmTMB parity matters.)
2. **Coupled two-axis RE syntax** — the defining feature is a *shared* `(1|species)`
   (or `phylo(1|species)`) on BOTH the `mu` and `sigma` formulas, tied by one 2×2 Λ.
   `_split_ranef` currently parses each formula independently; a rule is needed to
   recognise the same grouping on both axes and route it jointly.
3. **Accessor / `summary` names** for the 2×2 group-level covariance.

## Tree height (2026-10-17, #952 lane n5)

The routes here build Q from the raw branch lengths and start from absolute log SDs
(λ0 = [log 0.3, 0, log 0.3]). On ade4::carni70 (ultrametric, height 53.8), the
coupled Gamma location-scale fit `(1 | p | phylo(species))` on both `mu` and
`sigma` returns logLik = -1e18 at the start values, not converged, with no warning.
On the same tree divided by its height it fits (-187.3208, at the cor = 1 boundary).
Evidence: `docs/dev-log/evidence/952-sigma-phylo-tree-scale/survey/gamma_phylo_sigma/`.
NB2 shares the start on raw Q but was not measured.

The Gaussian sigma-phylo route (`_fit_gaussian_locscale_phylo`, a separate engine)
had the same failure. It now fits on the unit-height tree and reports on the raw
scale (#952). The same rescaling is the proposed fix for `_fit_locscale`.

## Known limitations

- One grouping factor per fit (the augmented state is single-grouping). Crossed
  *two-axis* groupings would need a multi-grouping augmented state.
- Inference is Wald only (no profile / bootstrap for this model yet).
- Variance-component recovery is good but, like all Laplace fits, the scale-axis RE
  is the harder axis to identify (needs enough obs per group).
