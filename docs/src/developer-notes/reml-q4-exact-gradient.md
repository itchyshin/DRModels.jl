# The exact gradient of the q=4 REML objective

Developer note for `src/reml_q4.jl`. Written **before** the implementation
(symbolic-alignment discipline): the maths is fixed here first, and every term
is then mapped, term-by-term, onto a named quantity the code already computes
or a named quantity to build.

Motivating defect: `fit_q4_reml` certified convergence on a **central
finite-difference** gradient with step `h_inner = max(h_fd, 5e-4)`
(`src/reml_q4.jl:420`, `fg!` at `:422–460`). On the `biv-q4-phylo-reml`
fixture the run-to-run noise of the objective is ~1e-3, i.e. the same order as
`g_tol = 1e-3`, so the certification was at or below the FD/noise floor. Every
basin/polish strategy tried against that certification plateaued
(`scratchpad/p12a-summary.md`, `p12a-basin-summary.md`).

---

## 1. The objective as implemented

`reml_ll_and_mode` (`src/reml_q4.jl:163–284`) returns, at outer parameters
`φ = (β_ρ[kr], lc[10])`:

| line | code | symbol |
|---|---|---|
| `:167` | `Lam = lc_to_Λ(lc)` | `Λ(lc)` — 4×4 among-axis covariance, log-Cholesky parameterised |
| `:168` | `P = prior_precision(Q_cond, inv(Lam))` | `P(lc) = Q ⊗ Λ⁻¹`, node-major, sparse |
| `:195–218` | alternation `estep_mode` / `cond_newton_beta` | joint profiled mode `(û, β̂)` |
| `:226` | `ml_ll = laplace_ll(...)` | `ℓ_ML = −J(û,β̂;φ) − ½ logdet H_uu + ½ logdet P` |
| `:247–266` | `H_u_beta`, `H_beta_beta` | `B = H_uβ`, `D = H_ββ` |
| `:269–273` | `C = ch_H \ H_u_beta`; `S = D − B'C` | `C = A⁻¹B`, `S = D − B'A⁻¹B` |
| `:281` | `ld_S = logdet(ch_S)` | `logdet S` |
| `:283` | `ml_ll - 0.5*ld_S` | `L_REML(φ)` |

with `A ≡ H_uu` (the CHOLMOD factor `ch_H` from `estep_mode`), and
`_reml_normalise` (`:302`) adding the constant `(n_β/2)·log 2π`, which has zero
gradient and is ignored below.

### 1.1 The one identity the whole derivation rests on

`H_uβ` and `H_ββ` are built at `:247–266` from the **same** per-leaf 4×4
Hessian `Hb = leaf_hess(...)` as `H_uu`'s leaf blocks, because every axis's
fixed effect enters the leaf through `η_d + u_d` (the header comment at
`:229–235`). Writing, for data row `i` at leaf node `t` (`base = 4(t−1)`), the
**lift**

```
R_i : z = (u, β) ↦ 4-vector of axis totals,   R_i = [E_i | F_i]
E_i[d, base+d] = 1                (u part, 4 × n_u)
F_i[d, off[d]+k] = X_d[i,k]       (β part, 4 × n_β), X_1..X_4 = X1, X2, Xs1, Xs2
```

the leaf's entire contribution to the joint Hessian over `z` is `R_iᵀ Hb R_i`.
`off` and the per-axis designs are literally `off`/`Xax` at `:239–242`.

Therefore the **joint Hessian over the augmented state** `z = (u, β)` is

```
𝓗 = [ A   B ]      A = H_uu,  B = H_uβ,  D = H_ββ
    [ B'  D ]      logdet 𝓗 = logdet A + logdet S
```

and the implemented objective collapses to a single Laplace form:

```
L_REML(φ) = −J(ẑ; φ) − ½ logdet 𝓗(ẑ; φ) + ½ logdet P(φ)          (★)
```

`J(z;φ)` is `joint_nll` with `β` promoted from a fixed argument to part of the
state, and `½ logdet P` is unchanged (the flat prior on `β` contributes no
`logdet`). **(★) is structurally identical to the ML objective in
`fit_q4_sparse_tmb.jl:10–15`, with `u → z` and `H → 𝓗`.** That is what makes
the existing exact-gradient machinery reusable term-for-term.

---

## 2. The gradient

At the joint stationary point `∇_z J(ẑ;φ) = 0`, the implicit-function
derivative of (★) is the exact analogue of
`marginal_and_exact_grad` (`src/fit_q4_sparse_tmb.jl:288–434`). For the
minimised objective `L⁻ = −L_REML` (what LBFGS sees, divided by `nobs`):

```
dL⁻/dφ_k = ∂/∂φ_k [ J + ½ logdet 𝓗 − ½ logdet P ]|_{ẑ frozen}      (CHEAP)
           − ∂/∂φ_k [ (∇_z J)ᵀ w ]|_{ẑ, w frozen}                   (IMPLICIT)
w = 𝓗⁻¹ v,   v = ½ ∇_z logdet 𝓗
```

### 2.1 Selected inverse of 𝓗 without factorising 𝓗

Never form or factor `𝓗`. With `C = A⁻¹B` (already computed, `:269–272`) and
`S⁻¹` (already factored, `ch_S` at `:275`):

```
𝓗⁻¹ = [ A⁻¹ + C S⁻¹ Cᵀ   −C S⁻¹ ]
      [ −S⁻¹ Cᵀ            S⁻¹   ]
```

Every entry needed is `A⁻¹` at the Takahashi pattern (`takahashi_selinv(ch_H)`,
`src/takahashi_selinv.jl`, O(p)) plus a rank-`n_β` correction with `n_β = 6` on
this fixture. Cost stays O(p·n_β²).

The single object each leaf needs is the 4×4

```
Ω_i = R_i 𝓗⁻¹ R_iᵀ = Vsel[base+1:base+4, base+1:base+4] + G_i S⁻¹ G_iᵀ
G_i = C[base+1:base+4, :] − F_i          (4 × n_β)
```

which is the REML replacement for the ML code's `Vblk`
(`fit_q4_sparse_tmb.jl:343`). **`Ω_i = Vblk` exactly when `F_i = 0` and
`S⁻¹ = 0`, i.e. when nothing is marginalised — the ML reduction.**

### 2.2 Alignment table — every term, its symbol, its code home, its scale

| # | Term | Symbol | Scale / parameterisation | Code home |
|---|---|---|---|---|
| C1 | `∇_φ J(ẑ;φ)` | `∂J/∂β_ρ`, `∂J/∂lc` | `β_ρ` identity link (`ηr = Xr·β_ρ`, `ρ = RHO_GUARD·tanh ηr`); `lc` log-Cholesky | **to build**: `reml_joint_nll_T(prob, P_t, û, β̂_t)` — copy of `joint_nll_T` (`fit_q4_sparse_tmb.jl:442`) with `β.rho` type-generic; single-level ForwardDiff over `φ`, mirrors `jn_of_θ` (`:319–325`) |
| C2 | `−½ ∇_lc logdet P = +½·N·∇_lc logdet Λ` | `glogdetΛ` | log-Cholesky; `N = prob.n_total` | **exists verbatim**: `fit_q4_sparse_tmb.jl:330–332` |
| C3a | `½ tr(𝓗⁻¹ ∂𝓗/∂β_ρ)` | `sη[5]` via `Ω_i` | identity link on `β_ρ`, chained through `Xr[i,:]` | **adapt**: `fit_q4_sparse_tmb.jl:340–367`, replacing `Vblk → Ω_i`; only the `m = 5` (`ηr`) column is needed, since `β_μ`/`β_σ` are no longer outer parameters |
| C3b | `½ tr(𝓗⁻¹ ∂𝓗/∂lc_k)` | `Gst` ⊗ `Mk = −Λ⁻¹(∂Λ/∂lc_k)Λ⁻¹` | log-Cholesky | **adapt**: `fit_q4_sparse_tmb.jl:374–396`; `Gst` accumulates `W_uu = A⁻¹ + C S⁻¹ Cᵀ` at the `Q_cond` pattern instead of bare `Vsel` — one extra `q·(C[bt+a,:]ᵀ S⁻¹ C[bs+b,:])` per nonzero |
| I1 | `v = ½ ∇_z logdet 𝓗` | `v = (v_u, v_β)` | natural `z` scale | **adapt**: `fit_q4_sparse_tmb.jl:402–418`. Per leaf, `ṽ_c = ½ Σ_{a,b} Ω_i[a,b]·T[a,b,c]` with `T = leaf_hess_du` (`:264`); then `v[base+c] += ṽ_c` **and** `v_β[off[c]+k] += ṽ_c·X_c[i,k]` (the new β rows, from `R_iᵀ`) |
| I2 | `w = 𝓗⁻¹ v` | `w = (w_u, w_β)` | — | **to build**: block solve. `q = S⁻¹(Cᵀv_u − v_β)`; `w_u = A⁻¹v_u + C q` (one `ch_H \`), `w_β = −q`. No 𝓗 factorisation |
| I3 | `−∇_φ[(∇_z J)ᵀ w]` | — | `φ` scale | **to build**: `scalar_of_φ` mirroring `fit_q4_sparse_tmb.jl:424–430`; `∇_z J` = `joint_grad_T` for the `u` block **plus** the β block `∂J/∂β_d[k] = Σ_i leaf_grad(...)[d]·X_d[i,k]` |
| N | `(n_β/2)·log 2π` | — | constant | `_reml_normalise` (`:302`) — **zero gradient** |
| Sc | objective scaling | `/nobs` | `nobs = length(prob.leaf_node)` | matches `neg_reml` (`:414`), so the returned gradient must be divided by `nobs` too |

### 2.3 Where the FD gradient was wrong, precisely

The FD gradient perturbs `φ_k` by `5e-4` and **re-runs the whole alternation**
(`fg!`, `:435–443`), so each one-sided evaluation carries the alternation's own
`1e-4·(1+‖β‖)` exit slack (`:216`) plus the mode-finder's residual. The exact
gradient (★) never re-solves: it evaluates one stationary point and applies the
implicit-function theorem, so its accuracy is set by `‖∇_z J(ẑ)‖`, not by a
difference of two noisy solves.

### 2.4 Consequence for certification

(★) is exact **only at a joint stationary point of `J` over `z = (u, β)`**. The
existing alternation exits on a relative β criterion (`:216`) that leaves
`‖∇_β J‖` at ~1e-4·‖β‖ — enough to poison an exact gradient certified at 1e-6.
The fix is already paid for: `𝓗` is exactly the joint Newton matrix, and §2.1
gives a block solve against it for free. A handful of joint Newton steps

```
z ← z − 𝓗⁻¹ ∇_z J(z)        (same block solve as I2)
```

drives `‖∇_z J‖` to ~1e-10, after which the exact gradient is trustworthy and
convergence can be certified on it. This is the "Newton-grade certification"
the FD route could not provide.

---

## 3. Found while implementing: the diagonal-Λ pattern degeneracy

`prior_precision(Q, Λ⁻¹)` (`src/sparse_aug_plsm.jl:79`) calls `sparse(Λinv)`,
which **drops zeros**. At an exactly diagonal `Λ` — including `fit_q4_reml`'s
own default warm start `Λ0 = 0.3I(4)` (`src/reml_q4.jl:374`) — the cross-axis
entries of `P`, and hence of `H_uu` at every non-leaf node, are *structurally*
absent. The Cholesky pattern shrinks with them, so the Takahashi selected
inverse cannot return `A⁻¹` at entries the `logdet H` trace genuinely needs.
They are missing, not zero.

Measured on `biv-q4-phylo-reml` at `Λ = 0.3I`, exact vs a step-scanned central
difference: two `lc` components were wrong by **0.037** and **13.70**; with a
`1e-6` off-diagonal added to `Λ`, **every** component agreed to **6.7e-8**.

Fix in the REML path (at the time): a local helper, `_reml_prior_precision`,
stored all 16 entries of the 4×4 axis block explicitly. The matrix was
numerically identical; only the pattern changed.

**`marginal_and_exact_grad` (`fit_q4_sparse_tmb.jl:374–384`) builds its `Gst`
from the same construction and therefore shares this degeneracy on the ML
path.** This was outside the scope of the original REML repair, but it should be checked: the
ML fit also starts at `Λ0 = 0.3I`, so its very first exact gradient is taken at
the degenerate point.

**Update:** `prior_precision` itself (`src/sparse_aug_plsm.jl`) now stores the
full q×q axis block unconditionally, so it no longer drops the cross-axis
entries at a diagonal `Λ`. That made the local `_reml_prior_precision` guard
redundant: the two builders produced bit-identical sparse output on this
fixture (same pattern, same `nzval`, `nnz` = 1376 at both a diagonal and a
non-diagonal `Λ`), and the redundant helper was removed while its call sites — including
`reml_ll_and_mode` (`src/reml_q4.jl:309`), whose call was previously the only
one left on the unguarded `prior_precision` — were repointed to
`prior_precision` directly.
Every call site in `src/reml_q4.jl` now uses the same, now-correct,
`prior_precision`.

---

## 4. API

The two module-level entry points the exact path exposes (unexported; called
by `fit_q4_reml`'s `fg!`). Both take the augmented problem, the root-conditioned
tree precision `Q_cond`, and the outer parameter vector `φ = (β_ρ, lc)`.

```@docs
DRModels.reml_nll_exact
DRModels.reml_nll_and_exact_grad
reml_objective_at
```
