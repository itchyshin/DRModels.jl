# Toward an O(p) sparse multi-component LSS engine

Developer note for the sparse multi-component LSS route, which lives in
`src/gaussian_sparse_lss.jl` (there is no separate
`gaussian_sparse_lss_multi.jl`). It was written as the symbolic-alignment pass
before any code existed; the route has since been implemented, wired into
`drm()` by the D-206 router (§1), and covered by the default test suite
(`test/test_lss_sparse_multi*.jl`, auto-discovered by `test/runtests.jl`) — see §9 for its
documented internals. Scope: extend `#551`'s single-phylo-component O(p) sparse engine
(`src/gaussian_sparse_lss.jl`) to **several** `sd()` submodels in one fit,
the way `#555`'s dense engine (`_fit_gaussian_lss_multi`,
`src/gaussian_lss.jl:564`) already does. Issue #563, Phase-2 slice S7b.

Reviewer contract item 6 (`docs/dev-log/plans/2026-08-30-julia-r-parity.md`,
"Reviewer-required contracts"): *"Shared component precision uses full
cross-component incidence products. Check nested/crossed groups, graph fill
and memory. No universal O(p) claim for arbitrary dense/crossed
structures."* This note's job is to measure that fill, not assume it.

---

## 1. The model

Per-observation Gaussian response with a heteroscedastic residual and `m`
independent variance components, each with its own log-SD submodel
(`src/gaussian_lss.jl:544–552`):

```
y_i = (Xμ βμ)_i + Σ_{c=1}^m (Z_c a_c)_i + e_i
e_i ~ N(0, σ_e,i²),        log σ_e,i = (Xσ βσ)_i
a_c ~ N(0, D_c K_c D_c),   log σ_c,g = (Z_c α_c)_g   (g = 1..G_c)
D_c = diag(σ_c,·) = diag(exp(Z_c α_c))
K_c = I_{G_c}                for an iid component
K_c = phylogenetic corr.     for the (at most one) phylogenetic component
```

Marginal, as already assembled by the dense route:

```
V = Σ_c Z_c D_c K_c D_c Z_cᵀ + diag(σ_e,i²)          (gaussian_lss.jl:548–552, 585–599)
```

**Log-SD combination is two different operations, and the code draws the
line at the component boundary:**

- *Within* one component, several predictor columns combine **additively on
  the log scale**, i.e. **multiplicatively on the SD scale**:
  `σ_c,g = exp(Z_c[g,:]·α_c) = Π_k exp(Z_c[g,k]·α_c,k)`
  (`gaussian_lss.jl:582`, `σa = exp.(c.Zg * α)`; the single-component sparse
  engine's `ησa = Zg * α; σa = exp.(ησa)` at `gaussian_sparse_lss.jl:50–52`
  is the same identity for one component).
- *Across* components the covariances **add**, not multiply — line
  `Vm[i, j] += Σc[...]` runs once per component inside the `for (ci, c) in
  enumerate(comps)` loop (`gaussian_lss.jl:576, 606`). This is the ordinary
  independent-variance-components model, not a product-of-SDs across
  components.

**IN SCOPE for a sparse extension:** one phylogenetic component (as #551)
plus any number of **iid** components, where every iid grouping is **nested**
in the sense §2 measures (a clean partition — no observation's group
membership is independent of the other components' membership). This is the
regime the augmented-tree-plus-hub construction stays close to O(p) in.

**OUT OF SCOPE, with reason:** two or more components whose group memberships
are mutually **crossed at comparable scale** (e.g. two ~200-level iid factors
with independent random assignment). §2 measures this filling catastrophically
regardless of elimination order — it is a structural property of the graph,
not an implementation gap. The existing **dense** multi-component route
already fits such models today (fixture B below is a real crossed example)
up to its `n ≤ 5000` cap (`gaussian_lss.jl:570`); this note is only about
whether an O(p) *sparse* alternative exists, and for that topology it does
not.

**S7b.4 router rule (D-206, `gaussian_lss.jl`'s `_drm_gaussian_lss_multi`).**
The public router turns the IN-SCOPE/OUT-OF-SCOPE classification above into a
decision at fit time, mechanically: dispatch to the sparse multi-component
route ONLY when (a) **exactly one** `sd()` component is phylogenetic, AND (b)
**every** iid component is either NESTED within it — `_lss_iid_nested_in`,
each iid group level co-occurs with exactly one phylo group level across all
rows, the same clean-partition test §2 uses — OR **small**: `G_c ≤
_LSS_SPARSE_SMALL_FRACTION × G_phy` with `_LSS_SPARSE_SMALL_FRACTION = 0.1`
(10% of the phylogenetic component's tip count), the documented threshold for
"`G_c ≪ p`" (§2.1 case (b), §7's S7b.2 estimate row) as a checkable rule
rather than a judgement call. Any iid component failing BOTH tests (crossed
at comparable scale, case (c)) routes the WHOLE model to dense, even under an
explicit `algorithm = :sparse` request — the router never silently runs an
unproven-fill sparse fit, and never silently drops the request either: an
explicit sparse request that falls back emits an `@info` naming which
component and why. `:auto` (the `drm()` default) auto-dispatches to sparse
when eligible AND the phylogenetic component has more than 500 tips,
mirroring the single-component `G > 500` rule (`gaussian_lss.jl:402`).
Acceptance: oracle 4 below.

---

## 2. The augmented precision

Stack the latent state `z = (a_phy; a_iid,1; …; a_iid,m)` (tree nodes `q =
2p-2`, `gaussian_sparse_lss.jl:19`, plus each iid component's `G_c` group
levels). The augmented precision generalises `H = Qs + Diagonal(diag_ZtWZ)`
(`gaussian_sparse_lss.jl:41, 63`) to a block matrix with genuine
**off-diagonal cross-component blocks**:

```
H = blockdiag(Q_phy, D_iid,1⁻¹, …, D_iid,m⁻¹) + Zᵀ W Z
Z = [Z_phy | Z_iid,1 | … | Z_iid,m]     (n rows; one leaf/group column-block per component)
```

`Zᵀ W Z`'s diagonal blocks are exactly `diag_ZtWZ` per component (each
observation touches exactly one leaf and one group per component, so those
blocks stay diagonal — no change from #551). The **cross blocks**
`(Z_k)ᵀ W Z_l`, `k ≠ l`, are new: entry `(t, g)` is `Σ_i w_i·[obs i has
component-k index t AND component-l index g]`. Their nonzero pattern is the
species↔group **incidence bipartite graph**, and that pattern — not the tree
— governs symbolic-Cholesky fill.

### 2.1 Measured fill (CHOLMOD `cholesky(Symmetric(H))`, default AMD ordering)

Probe: `probe_s7b_fill.jl` in the session scratchpad (log:
`probe_s7b_fill.log`, same directory), run time < 0.2 s total. `H` built with
random positive
weights (fill depends only on the *pattern*, not the values); `nnz(L)` from
`nnz(sparse(cholesky(Symmetric(H)).L))`.

| Case | Topology | dim(H) | nnz(H) | nnz(L) | ratio L/H | nnz(L)/dim |
|---|---|---|---|---|---|---|
| (a) | phylo(1000 tips) + iid(50), **nested** in species (partition) | 2048 | 8028 | 6213 | **0.774** | 3.03 |
| (b) | phylo(1000 tips) + iid(50), **crossed** with species | 2048 | 15578 | 23920 | **1.535** | 11.68 |
| (c) | two iid(200,200), fully **crossed**, n=5000, no phylo | 400 | 9848 | 56141 | **5.701** | 140.35 |

`(a)` vs `(b)` isolate one variable (nested vs. crossed group membership,
same sizes): crossing costs a **~3.85× increase in nnz(L)** (6213 → 23920),
but the result is still `< nnz(H)^{1.something}` and far from dense — because
one side of the incidence graph (the 50 groups) is small relative to the
tree (1998 nodes), so CHOLMOD can defer group elimination and absorb species
fill into their tree ancestors instead of forming a group-species clique.

Scaling sweep (`G2 = 50` fixed, `n = 5p`, `p ∈ {1000, 2000, 4000}` — same
script, log tail):

| p | nested nnz(L)/p | crossed nnz(L)/p |
|---|---|---|
| 1000 | 3.03 | 11.62 |
| 2000 | 3.02 | 11.46 |
| 4000 | 3.01 | 11.32 |

Both stay **flat as p grows with G2 fixed** — i.e. for *this* topology
(one large tree-structured component, one small fixed-size crossed factor)
the measured fill is O(p) empirically over this range, not just for the
nested case. **This does not extend to case (c):** `nnz(L)/dim = 140` on a
400×400 system (70% of the dense 400² entries) shows that when **both**
crossed factors are large (comparable size to each other, no small "hub"
side), no elimination order rescues sparsity — the classic two-way
crossed-random-effects fill result. Contract item 6's caution is confirmed
for that topology specifically, not for "any crossing."

### 2.2 Real fixture check

`probe_s7b_fixtureB.jl` (log: `probe_s7b_fixtureB.log`, both in the session
scratchpad) reruns the same
measurement on the **actual committed** `#555` fixture B design (32-tip tree,
16 studies genuinely crossed with species — verified in §5 of the recon:
each study spans ~12 species, each species appears in ~6 studies):

```
fixture B (real, crossed):        dim(H)=78  nnz(H)=582  nnz(L)=783  ratio=1.345
same sizes, nested comparator:    dim(H)=78  nnz(H)=262  nnz(L)=208  ratio=0.794
```

Same qualitative signature as the synthetic sweep (crossed ratio > 1, nested
< 1) at fixture scale — this is the "small crossed fixture" §5 uses as an
acceptance oracle.

### 2.3 When symbolic Cholesky stays O(p) — summary

- **Nested groups** (partition-compatible with the tree, `case (a)`): O(p),
  confirmed flat to p=4000.
- **One phylo component + a small crossed iid factor** (`case (b)`,
  `G_c ≪ p`): empirically near-linear over the measured range, but **not
  proven asymptotically** — G2 was held fixed at 50 while p grew; the
  argument for why (a hub-and-tree pattern lets AMD defer the small side)
  would need to be checked again if `G_c` also scales with `p`.
- **Two or more crossed components at comparable scale** (`case (c)`): **NOT
  O(p)** — fill is a fixed fraction of the dense matrix even at trivial
  size. **No universal O(p) claim** for this topology, as the contract
  requires.

**Caveat (added after adversarial re-probing, §8 finding 2 —
`review_s7b_fill2.jl/.log`): the flatness above was measured only under
uniform group sizes and a balanced tree.** Two further sweeps, same
in-scope nested topology (`G2 = 50`, balanced or caterpillar tree,
`n = 5p`):

- **Unbalanced group sizes** (90% of species in one giant group, the
  remaining 10% spread over the other 49): `nnz(L)/p` = 6.77 → 7.06 → 7.24 at
  p = 1000/2000/4000 — a real, if mild, **upward drift** (+7% over a 4×
  increase in `p`) that the uniform-size sweep never surfaces, and one that
  has not visibly asymptoted by p=4000. **What would settle it:** the same
  sweep carried to p=16,000–32,000 to see whether `nnz(L)/p` converges or
  keeps climbing.
- **Caterpillar tree** (`random_caterpillar_tree`, the worst case for
  CHOLMOD ordering — see its docstring) instead of `random_balanced_tree`,
  nested `G2=50`: `nnz(L)/p` stays flat (6.03/6.02/6.01 at p=1000/2000/4000)
  but at **~2× the balanced-tree constant** (≈3.0) — a level shift, not
  growth, so still O(p), but "flat to p=4000" was only ever demonstrated on
  one tree shape. **What would settle it:** repeat across the shape sweep
  #16 already uses elsewhere in this codebase (balanced, caterpillar, and at
  least one intermediate topology) before treating the constant as
  shape-independent.

Neither sweep breaks the O(p) classification for the in-scope regime; both
narrow what "confirmed" actually covers, which is why they are recorded here
rather than folded into the headline claim above.

---

## 3. The exact gradient — alignment table

Generalising #551's single-component machinery (`gaussian_sparse_lss.jl`) to
the block `H` of §2. Objective and logdet structure are unchanged in form —
`logdetV = logdetD + logdetCprior + logdetH` (`:85–87`) becomes
`logdetV = logdetD + Σ_c logdetCprior,c + logdetH` — but every per-leaf loop
that walks `1:G` (the single tree) must become a loop over **all component
nodes** (tree leaves ∪ every `Σ G_c`), and `wts[i]` generalises to one
scaling factor per component per observation.

| Symbol | Meaning | Existing code name (file:line) or TO BUILD | Fixture / oracle that pins it | Tolerance |
|---|---|---|---|---|
| `Q_c` | prior precision of component `c` (`Q_phy` from the tree; `D_iid,c⁻¹ = I` for iid, since `K_c=I`) | `Qs` (`gaussian_sparse_lss.jl:20`) for the phylo block; **TO BUILD**: trivial diagonal for each iid block | §5 oracle 1 (dense identity) | atol = 1e-5 (loglik) |
| `H` | augmented block precision, §2 | `H_template`/`H` (`:41, 63`) generalises to the block form of §2 — **TO BUILD**: multi-component assembly loop | §5 oracle 1 | rtol = 2e-4, atol = 2e-5 (coef) |
| `wts_g = σ_a,g · inv_sd_g` | the `D_a T` row-scaling: component SD × Takahashi-derived conditional SD (phylo) or component SD alone (iid, `inv_sd ≡ 1`) | `wts_g` (`:53`), `inv_sd` (`:27`) — generalise per component | §5 oracle 2 (FD grad) | grad ≤ 1e-6 |
| `b_c = Z_cᵀ W r` | per-component RHS for `â = H⁻¹b` | `b` accumulation loop (`:75–79`) — **TO BUILD**: one accumulation per component into the shared `b` vector at the component's block offset | §5 oracle 1 | rtol = 2e-4, atol = 2e-5 (coef) |
| `â = H⁻¹ b` | joint BLUP over all components | `ch \ b` (`:81`) — unchanged pattern, bigger `H` | §5 oracle 1 | rtol = 2e-4, atol = 2e-5 (coef) |
| `quad_sos` | zero-cancellation GMRF sum of squares | `dot(res_lat, w .* res_lat) + dot(â, Qs*â)` (`:89`) generalises to `+ Σ_c dot(â_c, Q_c â_c)` | §5 oracle 1 | atol = 1e-5 (loglik) |
| `Hinv` (selected inverse, **full pattern**, not just the diagonal) | O(p) exact `H⁻¹` entries at every pair the sparse pattern of `H`/`L` connects, any component — this now includes **cross-component** pairs `(g, t)`, `g` in one component's block and `t` in another's | `takahashi_selinv(ch)` (`:96`, `src/takahashi_selinv.jl:103`) — **works unchanged on the bigger `H`**: `takahashi_selinv` returns the selected inverse at every nonzero of the Cholesky factor's fill pattern, and every `H[g,t]` §2 introduces is by construction already inside that pattern (fill only ever *adds* to a matrix's own nonzero set, never removes from it) — so the cross entries needed below cost nothing extra to obtain | §5 oracle 2 | grad ≤ 1e-6 |
| `g_βμ` | mean-block gradient | `-(Xμ' * u)` (`:99`) — **unchanged**, `u` only depends on the joint residual | §5 oracle 2 | grad ≤ 1e-6 |
| `g_βσ` | residual-scale gradient | loop (`:101–111`) — ~~**unchanged**: `s_ii = Hinv[r_node, r_node]` still reads off the (bigger) selected inverse at the phylo-leaf node~~ **CORRECTED 2026-09-02** (implementation review, S7b.2): needs the SAME cross-component generalisation as `g_α,c`. `Vinv_ii = w_i − w_i²·zᵢᵀH⁻¹zᵢ`, and `zᵢ` (row `i` of the joint `Z`) has ONE nonzero per component — so the bilinear form is the FULL `Σ_{c,c'} wts_c[i]·wts_c'[i]·Hinv[node_c(i), node_c'(i)]` over every component pair touched by row `i` (own AND cross), not the single-node read `Hinv[r_node,r_node]` #551 uses when `m=1`. Verified against central-FD in `test/test_lss_sparse_multi_gradient.jl`: max relative error 1.37e-7 / 9.4e-10 / 2.7e-10 at the three oracle-2 points (dense optimum / perturbed / boundary) | §5 oracle 2 | grad ≤ 1e-6 |
| `g_α,c` | per-component α gradient, log-det trace term | **corrected formula — diagonal alone is WRONG.** `∂H[g,g]/∂α_c,g` (own-node term, as #551) **plus** `∂H[g,t]/∂α_c,g` for every node `t` of every *other* component sharing an observation with group `g` (exactly the cross blocks §2 introduces): `trace_term_g = Hinv[g,g]·∂H[g,g]/∂α_c,g + 2·Σ_{t≠g} Hinv[g,t]·∂H[g,t]/∂α_c,g`. Single-component #551 has no such `t` (its only cross partner is the tree, already folded into the diagonal via `leaf_pos`), so `:114–126`'s bare `s_node = Hinv[r_node,r_node]` is the `m=1` special case, not the general one. ~~`quad_term` (`u[i]*Zâ[i]` sum) is unaffected — it only ever needed the fitted `â`, not `Hinv`~~. **CORRECTED 2026-09-02** (implementation review, S7b.2): `quad_term` needs each component's OWN contribution `wts_c[i]·â[node_c(i)]`, NOT the joint `Zâ_i = Σ_c' wts_c'[i]·â[node(i,c')]` this note's `Zâ_i` (multi) row defines. Using the joint sum collapses every component's `quad_term` to the SAME scalar total whenever every component has a single-column `Zg` — each component's own indicator-sum then reduces to `Σ_i u_i·Zâ_i` over ALL rows regardless of which component is being differentiated, so distinct components cannot come out with distinct gradients, which is wrong. Verified against central-FD in `test/test_lss_sparse_multi_gradient.jl`: max relative error 1.37e-7 / 9.4e-10 / 2.7e-10 at the three oracle-2 points (dense optimum / perturbed / boundary). **TO BUILD** | §5 oracle 3 (cross-term gradcheck) | grad ≤ 1e-6 |
| cross-block `∂H[g,t]/∂α_c,g` accumulation | for every pair of components `(c, c')` and every observation `i` whose row touches both a node `g` of `c` and a node `t` of `c'`, the derivative of the *cross* entry `H[g,t] = Σ_i w_i·wts_c[i]·wts_{c'}[i]` w.r.t. `α_c,g` is `w_i·(∂wts_c[i]/∂α_c,g)·wts_{c'}[i]` — a new per-observation accumulation loop, one for every unordered component pair actually crossed by shared observations (at most `m(m-1)/2` loops, each `O(n)`) | **TO BUILD**: no #551 analogue exists (single component has only one factor in `wts`, so this derivative never arose) | §5 oracle 3 | grad ≤ 1e-6 |
| `Zâ_i` (multi) | fitted joint random effect at row `i` | `Zâ` (`:82`) generalises to `Σ_c wts_c[i]·â_c[node(i,c)]` | §5 oracle 1 | rtol = 2e-4, atol = 2e-5 (coef) |

Every row above resolves to an existing #551 line to **adapt** except the
block-assembly loops (`H`, `b`, cross-block `∂H/∂α` accumulation), which are
genuinely new code — they are marked **TO BUILD** rather than silently left
blank. The `g_α,c` row was caught wrong-as-first-drafted by the adversarial
review (§8, finding 4): a 6-node nested toy (`review_s7b_gradcheck.jl/.log`,
diagonal-only `wts_iid = [1,2]` groups nested in 4 phylo-like nodes) gives FD
`d(logdet H)/dα_iid[1] = 1.400168`; the diagonal-only formula returns
`2.669901` (**91% error**); the corrected formula with the cross term returns
`1.400168` (agrees to `3.4e-10`). The fix costs nothing extra to fetch
(`Hinv[g,t]` is already inside the Takahashi pattern, argued above), so this
is a specification bug, not a feasibility gap — but it is exactly the bug
symbolic-alignment exists to catch before code is written.

---

## 4. The REML correction

Both existing routes use the identical closed form — Patterson–Thompson,
one term regardless of component count:

```
L_REML = L_ML(θ̂) + ½ logdet(Xμᵀ V⁻¹ Xμ) − ½ pμ log 2π
```

Dense multi-component (`gaussian_lss.jl:634–638`): `A = Vfac \ Xμ; XtVinvX =
Xμ'*A; chX = cholesky(...); + 0.5*logdet(chX) - const_pμ` — one dense
`n×pμ` solve. The single-component sparse route computes the same quantity
via `pμ` sparse Cholesky **column** backsolves (`gaussian_sparse_lss.jl:135–
156`, `eval_reml`): for each column `k` of `Xμ`, accumulate `bX` at the
component's node positions and solve `ch \ bX`. **For the sparse multi
route, this generalises directly**: `bX` accumulates contributions from
*every* component's `wts` at that column, i.e. the same per-column loop but
summing `wts_c[i]*wXk[i]` into `bX` at each component's node for `i`'s row —
no new mathematics, just the same block-accumulation pattern as `b` in §3.
Cost stays `O(pμ · nnz(L))` sparse backsolves, same order as #551.

**Where the REML gradient actually lives (S7b.3).** The exact analytic
gradient of this correction — the Schur-complement identity `logdet(Xμ'V⁻¹Xμ)
= logdet(C) - logdet(H)` for the bordered matrix `C = [Xμ'WXμ Xμ'WZ; Zᵀ WXμ
H]`, and the selected-pairs matrix `Ψ[gi,gj] := dot(ÂX[gi,:], C2[gj,:])`
(`C2 = ÂX * (Xμ'V⁻¹Xμ)⁻¹`) that stands in for `Hinv[gi,gj]` in the REML
`g_α,c` trace term — is NOT re-derived in this note (an earlier draft left it
as a TO-BUILD placeholder here, S7b.2 found the omission): it is fully
derived and FD-verified in
[`DRModels._lss_sparse_multi_objective_and_grad`](@ref)'s own docstring
(`gaussian_sparse_lss.jl`, the function's REML paragraph), which is the
authoritative source for this derivation. Read it there rather than
re-deriving it from this section.

---

## 5. Acceptance oracles

1. **Dense-route identity, small nested fixture.** Fixture A
   (`test/fixtures/lsss/lsss_A.csv`, 32 species, `(1|species) +
   phylo(1|species)`, both `sd(species) ~ x`) — iid and phylo groupings
   **coincide** exactly (the strictest nested case). Assert the sparse
   multi-route's `loglik` matches `_fit_gaussian_lss_multi` to `atol = 1e-5`,
   both `α` blocks' coefficients to `rtol = 2e-4, atol = 2e-5` — the same
   dense-vs-sparse tolerances #551 itself ships with
   (`test/test_lss_sparse.jl:83–115`, `atol = 1e-5` on loglik/coef; the
   note's earlier `1e-8` draft was tighter than #551's own precedent with no
   measured margin to justify it, and was flagged as likely-flaky by the
   adversarial review, §8 finding 6).
   **WIRED (#563 S7b.4):** `test/test_lss_sparse_multi_public.jl` (public
   `drm(...; algorithm = :sparse)` route, not this specific CSV fixture —
   the S7b.1 nested fixture, 64 tips/192 sites, ML and REML; also covers the
   `G_phylo > 500` `:auto` dispatch case, S7b.4's own oracle 1c).
2. **FD-vs-exact gradient**, three points: the fixture-A MLE, a random
   interior point, and a boundary-ish point (one component's `α` pushed so
   `σ_c,g → e^{-6}`, mirroring #551's own boundary check). `‖g_exact −
   g_FD‖_∞ ≤ 1e-6` at all three — unchanged; this is the same bound #551
   uses for its own exact-vs-FD Hessian check (`gaussian_sparse_lss.jl:199–
   217`), not the looser `rtol=2e-4,atol=2e-5` line 198 uses for its packed
   *stored*-gradient regression test.
   **WIRED:** `test/test_lss_sparse_multi_gradient.jl` (S7b.2/S7b.2b, at the
   `_lss_sparse_multi_objective_and_grad` level) and
   `test/test_lss_sparse_multi_reml.jl` (S7b.3, the REML gradient); S7b.4
   only re-touches this indirectly through the public route's own
   convergence/gradient-norm checks (oracle 1c), it does not re-derive it.
3. **Cross-term gradcheck (finding 4, §3).** The generalisation of
   `review_s7b_gradcheck.jl`'s toy into a permanent unit test: a small nested
   multi-component fixture (phylo-like 4-node block + 2-level iid block,
   nested), assert the analytic `g_α,c` (with the cross `Hinv[g,t]` term)
   matches central-FD `d(logdet H)/dα` to `≤ 1e-6`, **and** assert the
   diagonal-only formula is rejected by the same test (`> 1e-3` away from
   FD) — a regression guard so the 91%-error version this note first drafted
   can never silently come back.
4. **Crossed fixture, fill prediction.** Fixture B (`lsss_B.csv`, real
   `study` crossed with `species`, §2.2) — assert the measured `nnz(L)/nnz(H)
   ≥ 1.2` (crossed) vs. the nested comparator's `< 1.0`, i.e. the note's §2.3
   claim is checked against the actual shipped fixture, not only synthetic
   data.
   **WIRED (router reading, #563 S7b.4):** `test/test_lss_sparse_multi_public.jl`
   exercises this at the ROUTER level instead of fixture B specifically — a
   genuinely crossed two-iid-component fixture (§1's router rule, D-206)
   asserts (i) `drm(...; algorithm = :sparse)` still returns the DENSE
   route's fit (via the `_lss_multi_route` marker) with an informative
   `@info`, never a silent sparse run on an unproven-fill topology, and (ii)
   the S7b.1 assembler's `H` on that same fixture measures `nnz(L)/dim > 4`
   — well above the nested band, matching this note's §2.1 case (c) reading.
   `test/test_lsss_multi.jl` separately pins fixture B itself on the DENSE
   route (unchanged by this sub-slice).
5. **Scaling to p = 10,000** — **only** if §2 continues to hold at that
   scale for the in-scope topology (one phylo + nested/small-crossed iid).
   This is a >30-minute class of run (D-139): pilot on Totoro first (≤150
   cores, per D-143), report the pilot's wall time and `nnz(L)/p` before
   committing to the full p=10,000 fit.
   **RUN (S7b.5, Totoro, `tools/lss_sparse_multi_scaling_pilot.jl`,
   `docs/dev-log/evidence/julia-r-parity/2026-09-02-lss-sparse-multi-
   scaling-pilot.md`):** the fill claim held (`nnz(L)/dim` flat, 3.085 →
   3.092 across the 1000 → 10,000 ladder) but wall time did NOT — log-log
   exponent ≈ 2.6–2.9 from p = 2500 → 10,000 (165 s at p = 10,000, RSS 1.3 →
   8.0 GB). **Root cause (S7b.6, this sub-slice):** not the sparse engine
   itself — `_drm_gaussian_lss_multi` (the public router,
   `src/gaussian_lss.jl`) called `_phylo_correlation(phy)`, a DENSE inverse
   of the ~2p×2p augmented tree matrix (`sigma_phy_dense`, cubic), **before**
   deciding `use_sparse`, on every fit with a phylogenetic `sd()` component
   — the sparse route never reads that matrix (it rebuilds a sparse
   precision from the tree directly), so the O(p³) cost was pure waste on
   every sparse-route call. Fixed by deferring materialisation with a
   `_LazyPhyloK()` marker (`src/gaussian_lss.jl`) until the dense route is
   actually selected.
   Local (laptop) post-fix re-measurement, same fixture/script, warm JIT at
   p = 200, single BLAS thread: p = 1000 → 0.17 s, p = 2000 → 0.30 s,
   p = 4000 → 1.55 s (down from the pre-fix router bug's p = 2000 → 3.77 s
   observed while diagnosing this sub-slice) — log-log exponent ≈ 1.6 from
   p = 1000 → 4000 (vs. ≈2.3–2.9 measured on the unfixed router at the same
   scale). Totoro re-measurement against the fix (head 86b7e3d4, same
   ladder, seeds and script, single-threaded): p = 1000 / 2500 / 5000 /
   10,000 → ML 8.7 (JIT) / 0.34 / 0.68 / 1.62 s, REML 0.76 / 0.94 / 1.55 /
   2.86 s; log-log exponent 2500 → 10,000 ≈ 1.1 (ML) / 0.8 (REML); RSS
   1.25–1.35 GB across the ladder (was 8.0 GB at p = 10,000); every
   log-likelihood bit-identical to the pre-fix run (the objective is
   untouched). Receipt: `docs/dev-log/evidence/julia-r-parity/
   2026-09-02-lss-sparse-multi-scaling-pilot.md`, "Post-fix re-measurement".

---

## 6. Memory bound

Bytes as a function of the measured `nnz`, using a standard CSC double
sparse-matrix cost `≈ 16·nnz + 8·(dim+1)` for `H`, and the same for the
Cholesky factor `L` (its own nnz):

| Case | dim | nnz(H) | bytes(H) | nnz(L) | bytes(L) |
|---|---|---|---|---|---|
| (a) nested, p=1000 | 2048 | 8028 | ≈145 KB | 6213 | ≈116 KB |
| (b) crossed, p=1000 | 2048 | 15578 | ≈265 KB | 23920 | ≈399 KB |
| (c) two crossed iid(200,200) | 400 | 9848 | ≈161 KB | 56141 | ≈901 KB |

For the in-scope regime (§2.3), `nnz(L) = O(p)` (empirically `≈3·p` nested,
`≈11.5·p` for one small crossed iid factor with `G_c=50`), so **memory stays
linear in `p`** for one phylo + bounded iid components. It is emphatically
**not** linear for two large crossed components — `(c)`'s 400×400 system
already needs `56,141` factor nonzeros, i.e. `nnz(L)/dim² ≈ 0.35`: memory
there scales like `O(G_1·G_2)`, the dense bound, well before `p` enters at
all.

---

## 7. Estimate and decision

Agent-hours for Phase 2, by sub-slice (nested + one-phylo scope only):

| Sub-slice | Work | Est. |
|---|---|---|
| S7b.1 | Block `H`/`b` assembly (multi-component, §2–3 TO BUILD rows) | 4–6h |
| S7b.2 | Per-component `g_α,c` gradient generalisation, incl. the own-node trace term and its FD check | 6–8h (raised from 3–4h — §8 finding 4/7: not a mechanical loop-widening; the diagonal-only formula this note first drafted was wrong (91% error vs FD), so the corrected term needs designing and checking, not just transcribing #551's loop) |
| S7b.2b | Cross-block `Hinv[g,t]`/`∂H[g,t]/∂α` accumulation — design + a permanent per-component-pair FD oracle (generalising `review_s7b_gradcheck.jl` into `test/`, §5 oracle 3) | 3–4h (new sub-slice; this logic has no #551 analogue to adapt) |
| S7b.3 | REML multi-component sparse backsolves (§4) | 2–3h |
| S7b.4 | Oracles 1, 2, 4 (§5), wired into `test/` | 3–4h |
| S7b.5 | p=10,000 pilot (Totoro, D-139) + scaling writeup, incl. the unbalanced/caterpillar follow-ups §2.3 leaves open | 2–3h (pilot only; full run separate) |
| **Total** | | **20–28h** |

**GO/NO-GO:** GO for the sparse multi-component engine restricted to one
phylogenetic component plus nested (or small, `G_c ≪ p`) iid components —
§2.1/§2.3 measure `nnz(L) = O(p)` and flat `nnz(L)/p` to p=4000 for both
sub-cases. NO-GO for a general "any crossed structures" sparse route — file
a separate issue if that is ever needed: case (c) measured `nnz(L)/dim² ≈
0.35` (near-dense) on a trivially small 400×400 system with two comparable-
size crossed factors, and no elimination order changes that. The existing
**dense** multi-component route (`gaussian_lss.jl:564`, `n ≤ 5000`) already
covers genuinely crossed models like fixture B and should stay the
recommended route for that topology.

**Limitations (S7b.6 update, this sub-slice).** The GO/NO-GO above is about
fill (`nnz(L)/p`), not wall time — the two turned out to be separate claims.
S7b.5's Totoro pilot measured fill flat but wall time superlinear (log-log
exponent ≈2.6–2.9, p=2500→10,000); S7b.6 traced that to a router bug (§5
oracle 5 above), not the sparse engine's own complexity, and fixed it. Local
(laptop) re-measurement to p=4000 dropped the exponent to ≈1.6, and the
Totoro re-run of the full ladder against the fix (§5 oracle 5) measured
p = 10,000 at 1.6 s (ML) / 2.9 s (REML) with exponent ≈ 1.1 / 0.8 from
p = 2500 → 10,000. The accurate summary is therefore **O(p) fill and O(p)
wall time to p = 10,000 on the nested fixture** (one phylogenetic component
plus one small iid component, n = 2p). Nothing here is measured for the
crossed layouts the router sends to the dense route, nor for many
components, nor beyond p = 10,000.

---

## 8. Adversarial review

1. **Fill claim (§2.1) re-run.** `probe_s7b_fill.jl`/`probe_s7b_fixtureB.jl` reproduced exactly:
   (a) 8028/6213, (b) 15578/23920, (c) 9848/56141; fixture B 582/783 vs 262/208 nested. No drift.
2. **Adversarial fill sweeps** (`review_s7b_fill2.jl/.log`): (i) `G_c` scaling with `p`
   (`G_c=p/10`, `p/2` at `p=1000,2000`) — flat holds, but at a **larger constant**
   (`nnz(L)/p ≈ 6.47`/`6.48` vs the note's `3.03` at `G_c=50`) — untested case, result is
   reassuring but the note never says the O(p) constant itself depends on `G_c`.
   (ii) **Unbalanced groups** (90% of species in one group, `G2=50`): `nnz(L)/p` = 6.77 → 7.06 →
   7.24 at p=1000/2000/4000 — a real, if mild (+7% over 4×p), **upward drift** the note's uniform-
   size measurements never surface; not yet visibly asymptoting at p=4000.
   (iii) **Caterpillar tree** (`random_caterpillar_tree`, CHOLMOD ordering stress), nested G2=50:
   flat and stable (6.03/6.02/6.01) but at **2× the balanced-tree constant** — untested topology,
   survives but the "flat to p=4000" headline was only ever measured on a balanced tree with
   uniform group sizes. Caterpillar+crossed also stays flat (~11–12, matching case (b)'s order).
   None of these break O(p); all narrow what "confirmed" actually covers.
3. **Model convention (§1).** `σa = exp.(c.Zg * α)` at `gaussian_lss.jl:582/613` confirmed —
   within-component predictors sum on the log scale as stated. The dense builder
   (`gaussian_lss.jl:690–739`) allows a general multi-column `Zg` per iid component and only ever
   pushes **one** `structured`/phylo block — the note's "(at most one) phylogenetic component,
   general iid predictor" scope is not a silent narrowing; it matches the builder exactly.
4. **Gradient table (§3), `g_α,c` row — WRONG as stated, load-bearing.** The claim "trace_term reads
   Hinv at that node" (diagonal only) is the single-component (#551) formula; it does not generalize.
   In the augmented joint state, `∂H/∂α_c,g` also perturbs the **cross-component blocks** `H[g,t]`
   for every other-component node `t` sharing an observation with group `g` (exactly the blocks §2
   itself introduces) — the trace needs `2·Σ_t Hinv[g,t]·∂H[g,t]/∂α_c,g` too. Numerically confirmed
   on a 6-node nested toy (`review_s7b_gradcheck.jl/.log`): FD = 1.400168; diagonal-only formula =
   2.669901 (**91% error**); formula with cross terms = 1.400168 (agrees to 3e-10). **Good news**:
   the needed `Hinv[g,t]` entries are always inside the Takahashi/`L` pattern for *any* topology
   (nested or crossed) — they equal exactly the pairs where `H` itself is already nonzero, and
   sparse-Cholesky fill always contains the original matrix's pattern. So this is a spec gap, not a
   feasibility gap — but the "TO BUILD" note for `g_α,c` must say so explicitly.
5. **REML (§4).** No bug: `bX` accumulation is linear per-component (no cross term), so `ch \ bX`
   over the full block `H` already captures cross-component coupling correctly through the solve.
   Aside: #551's current REML path uses `autodiff = :finite` (not the analytic ML gradient), so
   finding 4 does not currently corrupt REML — but a future analytic REML gradient would inherit it.
6. **Oracles (§5).** Tolerances are tighter than #551's own precedent: `test_lss_sparse.jl` uses
   `atol=1e-5` for dense-vs-sparse loglik/coef and `rtol=2e-4, atol=2e-5` for FD-vs-exact grad; the
   note proposes `1e-8`/`1e-6` with no stated reason to expect a tighter bound — likely flaky as
   written. Oracle 3's fixture-B fill assertion (`≥1.2` crossed vs `<1.0` nested) is confirmed with
   real margin (1.345 vs 0.794). p=10,000 correctly fenced behind D-139 (pilot first, not run).
7. **Estimate (§7).** #551 (single component, no cross-block bookkeeping) shipped 249 src lines +
   424 test lines in one PR with no itemized hour log, so no clean baseline exists — but given
   finding 4, S7b.2's 3–4h for "gradient generalization" undersells the work: it is not a mechanical
   loop-widening, it is new cross-component accumulation logic through Takahashi that has to be
   designed, tested per component-pair, and checked against FD (as done here) before it can be
   trusted.

VERDICT: APPROVE WITH REQUIRED EDITS
- Rewrite §3's `g_α,c` row to include the cross-component `Hinv[g,t]` trace terms (finding 4);
  state the general in-pattern guarantee so implementers don't think this needs re-deriving.
- Revise §5 oracle tolerances to match #551 precedent (`atol=1e-5` / `rtol=2e-4,atol=2e-5`) or justify tighter ones.
- Add a caveat to §2.3 that flatness was measured only under uniform group sizes and a balanced tree; cite the unbalanced-size drift and caterpillar-tree constant shift (finding 2) as open follow-ups.
- Revise S7b.2's estimate upward and add an explicit cross-block-accumulation design/test subtask.

Edits applied: 2026-09-02, by the author

Implementation review (S7b.2, 2026-09-02): two further table corrections applied — see §3 rows quad_term and g_βσ.

Implementation note (S7b.4, 2026-09-02): the public-route wiring added §1's
router rule and a `_lss_multi_route` marker (`gaussian_sparse_lss.jl`) so
tests can assert which engine a fit took — `DrmFit` (`gaussian_core.jl`) is a
plain positional struct with an 11-/19-/22-arg constructor ladder kept
stable specifically so ~70 call sites across ~20 family files never change
when a field is added, so this sub-slice deliberately did NOT add a `route`
field there; a side table keyed by `fit.theta` plays that role instead (NOT
by `fit` itself — `drm()` reconstructs a new immutable `DrmFit` via
`_withformula` on every route's return value, so the object the router marks
is never the object the caller gets back; `θ̂` is the one field every
`_with*` wrapper forwards by reference rather than copying, confirmed the
hard way when marking `fit` directly read back `:unknown` in this
sub-slice's own tests — see the marker's own docstring). Wiring
`algorithm`/`sparse` through to the
multi-component router also required one small forwarding edit at the
`drm()` dispatch call site (`gaussian_core.jl`, mirroring the identical
forwarding the sibling single-component phylo call two lines above already
does) — outside this sub-slice's originally scoped file list, but without it
an explicit `algorithm = :sparse` request on a multi-component `sd()` model
was silently dropped (never reaching the router at all), the same silent-
drop failure mode issue #2 named for σ-phylo; left in place rather than
leaving the feature unreachable from `drm()`.

## 9. API (internal, documented)

Module-level entry points of the sparse multi-component route (unexported; the public surface is `drm(...)` with several `sd()` parts).

```@docs
DRModels._fit_gaussian_lss_sparse_multi
DRModels._lss_sparse_multi_assemble
DRModels._lss_sparse_multi_objective
DRModels._lss_sparse_multi_objective_and_grad
DRModels._lss_sparse_multi_reml_pieces
DRModels._mark_lss_multi_route!
DRModels._lss_multi_route
DRModels._sparse_lss_iid_comp
DRModels._sparse_lss_phylo_comp
```
