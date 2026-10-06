# Cross-family dependence: statistical foundations

!!! note "Status — Experimental (engine), paper-section reference"
    The cross-family route is **Experimental** and not release-ready: the
    evidence is a set of simulated fixtures covering five family pairs, and
    there is no interval-coverage study. The
    maths below is what the code evaluates; the release status is what limits
    its use. See [Evidence & limits](../capabilities.md).
    This page is the methods reference for the cross-family bivariate model,
    with link-scale standardisation. It documents the
    shared-latent construction, the Gauss–Hermite marginal likelihood, the
    link-residual variances ``v_k``, the latent-scale correlation ``\rho`` and its
    identifiability, the three confidence intervals, and the exact reduction to
    the bivariate `rho12` Gaussian model. It is written to paper-section standard;
    every formula here is the one the code evaluates.

## 1. Motivation and scope

The bivariate Gaussian model (`bf(mu1=…, mu2=…, sigma1=…, sigma2=…, rho12=…)`)
couples two responses through a **residual correlation** ``\rho_{12}`` on a
common identity-link, real-valued scale. That construction is only available
when both responses are Gaussian: ``\rho_{12}`` is the correlation of two
residuals that live on the *same* scale, so it is undefined the moment one
response is a count, a proportion, or a strictly positive quantity.

The cross-family model removes that restriction. It couples two responses that
may come from **different** families — Gaussian × Poisson, Gamma × Beta,
negative-binomial × Binomial, and so on — through a single shared
per-observation latent variable, and reports the induced dependence as a
correlation on the latent (link) scale, ``\rho``. When both families happen to
be Gaussian the model reduces *exactly* to the `rho12` bivariate Gaussian
(Section 7), so it is a strict generalisation rather than a separate object.

Throughout, the two axes are indexed ``k \in \{1,2\}``; ``i = 1,\dots,n`` indexes
the ``n`` paired observations. We follow the package convention of naming the
scale ``\sigma`` (never ``\tau``). The residual-correlation symbol ``\rho_{12}``
is reserved for the same-scale Gaussian construction; the latent-scale
cross-family correlation is written ``\rho`` without a subscript.

## 2. The shared-latent construction

Each observation ``i`` carries one **scalar** standard-normal latent variable
``u_i``, shared by both responses. Conditional on ``u_i`` the two responses are
independent, each drawn from its own family with a family-specific linear
predictor:

```math
y_{1i} \mid u_i \sim \mathrm{fam}_1(\eta_{1i}), \qquad
y_{2i} \mid u_i \sim \mathrm{fam}_2(\eta_{2i}),
```
```math
\eta_{ki} \;=\; \mathbf{x}_{ki}^\top \boldsymbol{\beta}_k \;+\; \lambda_k\, u_i,
\qquad u_i \sim \mathcal{N}(0,1),\quad i = 1,\dots,n.
```

Here ``\mathbf{x}_{ki}^\top \boldsymbol{\beta}_k`` is the fixed-effect linear
predictor for axis ``k`` (design matrices ``X_k``, coefficients
``\boldsymbol{\beta}_k``), and ``\lambda_k`` is a scalar **factor loading** that
controls how strongly the shared latent feeds into axis ``k``. All dependence
between ``y_{1i}`` and ``y_{2i}`` flows through ``u_i``: it is a one-factor
model with a single common factor per observation. This is the canonical
generalised-linear latent-variable (GLLVM) device — a shared latent on the
linear predictor inducing cross-response dependence — specialised to two
responses and a single factor (Skrondal & Rabe-Hesketh 2004; Niku et al. 2019).

Each axis uses its family's canonical link, so ``\eta_{ki}`` is on the link
scale: identity for Gaussian, ``\log`` for Poisson / negative-binomial / Gamma,
``\mathrm{logit}`` for Binomial / Beta. The conditional mean on the response
scale is ``g_k^{-1}(\eta_{ki})`` (identity, ``\exp``, or logistic).

### Dispersion sub-model

Dispersion-carrying families also estimate a per-observation **log-native
dispersion** ``d_{ki}`` through its own linear predictor,

```math
\log d_{ki} \;=\; \mathbf{x}^{\sigma}_{ki}{}^{\!\top} \boldsymbol{\beta}^{\sigma}_k,
```

with design matrix ``X^{\sigma}_k`` (default a single intercept column, which
collapses ``d_{ki}`` to a constant). The natural dispersion ``d_{ki}`` plays the
family-specific role given in the table below. Poisson and Binomial are
*dispersionless* — they are pinned by ``\eta`` alone and carry no
``\boldsymbol{\beta}^{\sigma}_k``.

| Family | ``d`` is | Conditional density at node ``u`` |
|---|---|---|
| `Gaussian` | residual SD ``\sigma`` | ``\mathcal{N}(\eta,\sigma^2)`` |
| `Poisson` | — | ``\mathrm{Pois}(e^{\eta})`` |
| `Binomial` | — | ``\mathrm{Bin}(m,\,\mathrm{logistic}(\eta))`` |
| `NegBinomial2` | size ``\theta`` | ``\mathrm{NB2}(\mu=e^{\eta},\ \theta)``, ``\mathrm{Var}=\mu+\mu^2/\theta`` |
| `Beta` | SD ``\sigma``, precision ``\phi=\sigma^{-2}`` | ``\mathrm{Beta}(\mu\phi,(1-\mu)\phi)``, ``\mu=\mathrm{logistic}(\eta)`` |
| `Gamma` | SD ``\sigma`` (CV), shape ``\alpha=\sigma^{-2}`` | ``\mathrm{Gamma}(\alpha,\ \mu/\alpha)``, ``\mu=e^{\eta}`` |

The full parameter vector is

```math
\boldsymbol{\theta} = \bigl(\boldsymbol{\beta}_1,\ \boldsymbol{\beta}_2,\
\ell\lambda_1,\ \lambda_2,\ \boldsymbol{\beta}^{\sigma}_1,\ \boldsymbol{\beta}^{\sigma}_2\bigr),
```

where ``\ell\lambda_1`` is the log of the first loading (Section 5) and the
``\boldsymbol{\beta}^{\sigma}_k`` blocks are present only for dispersion-carrying
axes.

## 3. The Gauss–Hermite marginal likelihood

Because ``u_i`` is one-dimensional, the marginal likelihood is a product of
scalar integrals — one per observation — that we evaluate exactly (to quadrature
order) rather than by Laplace. Marginalising the shared latent,

```math
L(\boldsymbol{\theta})
= \prod_{i=1}^{n} \int_{-\infty}^{\infty}
   p_1\!\bigl(y_{1i}\mid \eta_{1i}(u)\bigr)\,
   p_2\!\bigl(y_{2i}\mid \eta_{2i}(u)\bigr)\,
   \phi(u)\,\mathrm{d}u,
```

with ``\eta_{ki}(u) = \mathbf{x}_{ki}^\top\boldsymbol{\beta}_k + \lambda_k u``
and ``\phi`` the standard normal density. The conditional log-densities
``\log p_k`` are the family terms tabulated in Section 2 (and coded in
`_mf_obs_ll`).

### Change of variables to Gauss–Hermite form

Gauss–Hermite quadrature approximates ``\int h(x)\,e^{-x^2}\,\mathrm{d}x \approx
\sum_{k=1}^{K} w_k\, h(z_k)`` with nodes/weights ``(z_k, w_k)`` (Golub–Welsch).
Writing ``\phi(u) = (2\pi)^{-1/2} e^{-u^2/2}`` and substituting
``u = \sqrt{2}\,z`` (so ``\mathrm{d}u = \sqrt{2}\,\mathrm{d}z`` and
``u^2/2 = z^2``) turns each observation's integral into

```math
\int p_{1i}\,p_{2i}\,\phi(u)\,\mathrm{d}u
= \frac{1}{\sqrt{\pi}} \int p_{1i}(\sqrt2\,z)\,p_{2i}(\sqrt2\,z)\, e^{-z^2}\,\mathrm{d}z
\;\approx\; \frac{1}{\sqrt{\pi}} \sum_{k=1}^{K} w_k\,
  p_{1i}(\sqrt2\,z_k)\, p_{2i}(\sqrt2\,z_k).
```

The ``1/\sqrt{\pi}`` prefactor is exactly the
``\texttt{half\_log\_pi} = \tfrac12\log\pi`` correction subtracted per
observation in the code. The negative marginal log-likelihood is therefore

```math
-\log L(\boldsymbol{\theta})
= -\sum_{i=1}^{n}\Biggl[
  \operatorname*{logsumexp}_{k}\Bigl(\log w_k
    + \log p_{1i}(\sqrt2\,z_k) + \log p_{2i}(\sqrt2\,z_k)\Bigr)
  - \tfrac12\log\pi
\Biggr],
```

evaluated with the numerically stable ``\operatorname{logsumexp}`` (subtract the
per-observation maximum, exponentiate, sum, re-log). The default node count is
``K = 32``.

### Why quadrature, not Laplace

For this model the integrand is genuinely one-dimensional, so a ``K``-node
Gauss–Hermite rule is both cheap (``O(nK)`` per likelihood evaluation) and far
more accurate than a single mode-plus-curvature Laplace match — it captures
skewed and heavy-tailed conditional posteriors that Laplace cannot see (the
shape/dispersion bias documented in
[Laplace vs variational marginals](marginal-la-vs-va.md)). The summand is a
smooth, deterministic function of ``\boldsymbol{\theta}`` (the nodes are
constants), so the objective is differentiable by forward-mode automatic
differentiation and is optimised with L-BFGS. A finite-penalty guard maps any
non-finite objective value (e.g. an overflow from a large Poisson loading
mid-line-search) to a large finite number of the same dual type, so the
optimiser backtracks rather than aborting; the well-conditioned path — hence
every converged estimate and interval — is untouched.

## 4. Link-residual standardisation

The loadings ``\lambda_k`` and the marginal log-likelihood are well defined, but
``\lambda_1`` and ``\lambda_2`` are not directly comparable across families: a
loading of ``0.5`` means something different on a ``\log`` scale (Poisson) than
on a ``\mathrm{logit}`` scale (Binomial) than on an identity scale (Gaussian).
To report a single interpretable dependence we standardise to the **latent
(link) scale**, following Nakagawa & Schielzeth (2010).

The key quantity is the family's own **observation-level variance on its link
scale**, the *distribution-specific variance* ``v_k``. On the link scale the
total latent variance of axis ``k`` decomposes as

```math
\underbrace{\operatorname{Var}(\eta_{ki})}_{\lambda_k^2}
\;+\;
\underbrace{v_k}_{\text{link-scale observation variance}},
```

so the latent-scale variance of axis ``k`` is ``\lambda_k^2 + v_k`` and the
cross-axis latent covariance is ``\lambda_1\lambda_2`` (the only shared term).
``v_k`` is computed by `link_residual` and is **reporting-only**: it never enters
the fit objective.

### Per-family ``v_k`` derivations

The link-residual map is exactly Nakagawa & Schielzeth's (2010, Table 2; see also
Nakagawa, Johnson & Schielzeth 2017) distribution-specific variance for each
family's canonical link, evaluated at a representative fitted mean
``\bar\mu_k = n^{-1}\sum_i g_k^{-1}(\eta_{ki})``.

| Family | Link | ``v_k`` | Origin |
|---|---|---|---|
| `Gaussian` | identity | ``\sigma_k^2`` | residual variance enters the latent scale directly |
| `Poisson` | ``\log`` | ``\log\!\bigl(1 + 1/\bar\mu_k\bigr)`` | delta-method / lognormal variance of ``\log y`` |
| `Binomial` | ``\mathrm{logit}`` | ``\pi^2/3`` | variance of a standard logistic; distribution-free |
| `NegBinomial2` | ``\log`` | ``\psi_1(\theta_k)`` | lognormal-approx. dispersion term, ``\psi_1`` = trigamma |
| `Gamma` | ``\log`` | ``\psi_1(1/\sigma_k^2) = \psi_1(\alpha_k)`` | variance of ``\log y`` under ``\mathrm{Gamma}(\alpha,\cdot)`` |
| `Beta` | ``\mathrm{logit}`` | ``\psi_1(\bar\mu_k\phi_k) + \psi_1\bigl((1-\bar\mu_k)\phi_k\bigr)`` | variance of ``\mathrm{logit}(y)``, ``\phi_k = \sigma_k^{-2}`` |

A few derivations worth spelling out, because they are the ones a reader will
want to check:

- **Gaussian, ``v_k = \sigma_k^2``.** With an identity link the residual *is* on
  the link scale, so its variance enters ``\lambda_k^2 + v_k`` directly. This
  differs from the GLLVM / gllvmTMB convention, which reports ``v = 0`` for a
  Gaussian trait because there the residual variance lives in a separate
  ``\boldsymbol{\Psi}`` covariance block. In this shared-latent
  parameterisation there is no separate ``\boldsymbol{\Psi}``: setting
  ``v = 0`` would force ``\rho = 1`` for Gaussian × Gaussian. ``v_k = \sigma_k^2``
  is the value verified by the exact Gaussian × Gaussian ≡ `rho12` parity
  (Section 7).

- **Poisson, ``v_k = \log(1 + 1/\bar\mu)``.** Under the ``\log`` link a Poisson
  mean ``\mu`` with ``\operatorname{Var}(y)=\mu`` has, to the lognormal /
  delta-method order used throughout Nakagawa & Schielzeth, an observation-level
  variance on the log scale of ``\operatorname{Var}(\log y)\approx
  \operatorname{Var}(y)/\mu^2 = 1/\mu``, refined to ``\log(1+1/\mu)`` (the exact
  lognormal-matching expression that reproduces ``1/\mu`` for large ``\mu`` while
  staying finite and positive for small ``\mu``). It is evaluated at the
  representative mean ``\bar\mu``.

- **Binomial, ``v_k = \pi^2/3``.** On the ``\mathrm{logit}`` link the observation
  variance is taken as the variance of the standard logistic distribution,
  ``\pi^2/3``. This is the standard latent-threshold value (the variance of the
  logistic latent in a logit model) and is distribution-free — it does not depend
  on ``\bar\mu`` or any dispersion.

- **Gamma, ``v_k = \psi_1(1/\sigma_k^2)``.** With a ``\log`` link and shape
  ``\alpha = 1/\sigma_k^2``, ``\log y`` for ``y\sim\mathrm{Gamma}(\alpha,\cdot)``
  has variance ``\psi_1(\alpha)`` (the trigamma function ``\psi_1 = \psi'`` is
  exactly the variance of ``\log`` of a Gamma deviate). Here ``\sigma_k^2`` is the
  squared coefficient of variation, the package's Gamma `sigma`-slot convention.

- **NegBinomial2, ``v_k = \psi_1(\theta_k)``.** The lognormal-approximation
  distribution-specific variance contributed by the size/dispersion ``\theta`` on
  the ``\log`` link, matching gllvmTMB's `nbinom2` entry.

- **Beta, ``v_k = \psi_1(\bar\mu\phi) + \psi_1((1-\bar\mu)\phi)``.** For
  ``y\sim\mathrm{Beta}(\mu\phi,(1-\mu)\phi)`` the variance of ``\mathrm{logit}(y)
  = \log y - \log(1-y)`` is the sum of the trigamma terms of the two Beta shape
  parameters (precision ``\phi = \sigma_k^{-2}``), evaluated at ``\bar\mu``.

## 5. The latent-scale correlation and identifiability

With the latent variances ``\lambda_k^2 + v_k`` and the shared covariance
``\lambda_1\lambda_2`` in hand, the **latent-scale correlation** is the natural
ratio:

```math
\boxed{\;\rho \;=\; \dfrac{\lambda_1\,\lambda_2}{\sqrt{(\lambda_1^2 + v_1)\,(\lambda_2^2 + v_2)}}\;}
```

This is the cross-family generalisation of a residual correlation: it is the
correlation of the two latent linear predictors after each axis is standardised
by its own total link-scale variance. ``\rho \in (-1, 1)`` by construction
(Cauchy–Schwarz, since ``v_k > 0``), with the sign carried by ``\lambda_2``.

### Identifiability

Three identifiability facts govern how ``\rho`` is parameterised and what is
estimable:

1. **Sign of the shared factor (``u \to -u``).** The likelihood is invariant
   under flipping the latent, ``u_i \mapsto -u_i``, which sends
   ``(\lambda_1,\lambda_2)\mapsto(-\lambda_1,-\lambda_2)`` and leaves both
   ``\eta`` distributions unchanged. This is the familiar factor-loading sign
   indeterminacy. We break it by constraining the **first loading positive**,

   ```math
   \lambda_1 = \exp(\ell\lambda_1) > 0,
   ```

   so ``\ell\lambda_1`` is unconstrained and ``\lambda_1`` cannot change sign.
   With ``\lambda_1 > 0`` pinned, the sign of ``\rho`` is the sign of
   ``\lambda_2``, which is now identified.

2. **Gaussian × non-Gaussian: fully identified.** When at least one axis is
   non-Gaussian, that axis has no free residual variance competing with the
   loading on the same scale (its observation variance ``v_k`` is fixed by the
   mean / dispersion, not by ``\lambda_k``). All parameters —
   ``\boldsymbol{\beta}_k``, both loadings, the dispersions, and hence ``\rho`` —
   are then separately identified.

3. **Gaussian × Gaussian: only ``\log L`` and ``\rho`` identified.** When *both*
   axes are Gaussian the model has a flat ridge in the individual loadings: the
   marginal is exactly bivariate normal (Section 7), and a bivariate normal is
   fully described by its means, the two marginal variances
   ``\lambda_k^2 + \sigma_k^2``, and the covariance ``\lambda_1\lambda_2``. The
   split of ``\lambda_k^2 + \sigma_k^2`` between the loading ``\lambda_k`` and the
   residual ``\sigma_k`` is **not** separately identified (any
   ``(\lambda_k, \sigma_k)`` on the ridge gives the same marginal variance). What
   *is* identified is precisely the marginal-covariance summary: the
   log-likelihood and the correlation ``\rho``. This is not a defect — ``\rho`` is
   the only cross-axis quantity the data can speak to in the Gaussian × Gaussian
   case, and it is exactly the bivariate `rho12` (Section 7). Reported individual
   loadings on this ridge should not be over-interpreted; the log-likelihood and
   ``\rho`` are the trustworthy outputs.

## 6. Confidence intervals for ``\rho``

Three intervals for ``\rho`` are provided, in increasing order of cost and
robustness. All operate on Fisher's ``z = \operatorname{atanh}\rho`` where a
transformation is used, because the sampling distribution of a correlation is far
closer to normal on the ``z`` scale.

### 6.1 Fisher-``z`` Wald (delta-method)

The default interval. Let ``\hat{\boldsymbol{\theta}}`` be the maximum-likelihood
estimate and ``\hat V = H^{-1}`` the inverse observed-information matrix, where
``H = \nabla^2 (-\log L)`` is the Hessian of the negative log-likelihood at
``\hat{\boldsymbol{\theta}}`` (computed by automatic differentiation). Define
``z(\boldsymbol{\theta}) = \operatorname{atanh}\rho(\boldsymbol{\theta})`` and its
gradient ``\mathbf{g} = \nabla_{\boldsymbol{\theta}}\, z(\hat{\boldsymbol{\theta}})``
(also by automatic differentiation through ``\rho`` and ``\operatorname{atanh}``).
The delta-method variance of ``z`` is

```math
\widehat{\operatorname{Var}}(\hat z) = \mathbf{g}^\top \hat V \mathbf{g},
\qquad \mathrm{SE}(\hat z) = \sqrt{\mathbf{g}^\top \hat V \mathbf{g}},
```

and the ``100(1-\alpha)\%`` interval is formed on the ``z`` scale and mapped back
by ``\tanh``:

```math
\Bigl(\ \tanh\bigl(\hat z - z_{1-\alpha/2}\,\mathrm{SE}(\hat z)\bigr),\ \
        \tanh\bigl(\hat z + z_{1-\alpha/2}\,\mathrm{SE}(\hat z)\bigr)\ \Bigr),
\qquad \hat z = \operatorname{atanh}\hat\rho,
```

with ``z_{1-\alpha/2}`` the standard normal quantile. The ``\tanh`` back-transform
guarantees the endpoints stay in ``(-1, 1)``. If ``H`` is not invertible or
``\mathbf{g}^\top \hat V \mathbf{g}`` is non-positive, the interval is returned as
`(NaN, NaN)` rather than a spurious number — the same honesty discipline the
engine applies whenever the observed information is singular.

### 6.2 Profile likelihood

The recommended interval when the Wald approximation is doubted (small ``n``, or
``\rho`` near the boundary), better calibrated than Wald and cheaper than the
bootstrap. Fix the correlation at a trial value ``\rho_0`` and re-optimise all
other parameters subject to that constraint, enforced with a quadratic penalty on
the ``z`` scale:

```math
\tilde{\boldsymbol{\theta}}(\rho_0) = \arg\min_{\boldsymbol{\theta}}\
\Bigl[\, -\log L(\boldsymbol{\theta})
   + c\,\bigl(\operatorname{atanh}\rho(\boldsymbol{\theta}) - \operatorname{atanh}\rho_0\bigr)^2 \Bigr],
\qquad c = 10^4 .
```

The profile deviance at ``\rho_0`` is

```math
D(\rho_0) = 2\Bigl[\, -\log L\bigl(\tilde{\boldsymbol{\theta}}(\rho_0)\bigr)
   - \bigl(-\log L(\hat{\boldsymbol{\theta}})\bigr) \Bigr],
```

and the profile-likelihood interval is the set
``\{\rho_0 : D(\rho_0) \le \chi^2_{1,\,1-\alpha}\}``, i.e. the two ``\rho_0`` where
``D(\rho_0)`` crosses the ``\chi^2_1`` quantile at level ``1-\alpha``. The two
endpoints are found by bisection on each side of ``\hat\rho`` (within ``\pm
0.999``); a side that never reaches the threshold inside ``(-0.999, 0.999)`` is
reported at that boundary.

### 6.3 Parametric bootstrap

The most robust and most expensive interval, useful as an external check on the
two analytic intervals. At the fitted ``\hat{\boldsymbol{\theta}}`` simulate
``B`` replicate data sets *from the model itself* — draw a fresh shared latent
``u_i^{(b)} \sim \mathcal{N}(0,1)`` per observation, form
``\eta_{ki}^{(b)} = \mathbf{x}_{ki}^\top\hat{\boldsymbol{\beta}}_k +
\hat\lambda_k u_i^{(b)}``, and draw ``y_{ki}^{(b)}`` from the corresponding
family at ``(\eta_{ki}^{(b)},\hat d_{ki})`` (the per-family samplers in
`_mf_rand`) — refit the model to each replicate, and collect the bootstrap
correlations ``\{\hat\rho^{(b)}\}_{b=1}^{B}``. The interval is the percentile
interval

```math
\Bigl(\ \hat\rho^{(b)}_{(\alpha/2)},\ \ \hat\rho^{(b)}_{(1-\alpha/2)}\ \Bigr),
```

the empirical ``\alpha/2`` and ``1-\alpha/2`` quantiles of the bootstrap
distribution. Non-converged replicates are discarded; the interval is returned
only if at least ``\max(10,\,B/2)`` replicates succeed, otherwise `(NaN, NaN)`.

## 7. Exact reduction to the `rho12` bivariate model

When both axes are Gaussian the cross-family model is **not** an approximation of
the bivariate Gaussian — it is algebraically identical to it, and ``\rho``
equals the residual correlation ``\rho_{12}``.

Take ``\mathrm{fam}_1 = \mathrm{fam}_2 = \texttt{Gaussian}`` with residual SDs
``\sigma_1, \sigma_2``. Conditional on ``u_i``,

```math
y_{ki} \mid u_i \sim \mathcal{N}\bigl(\mathbf{x}_{ki}^\top\boldsymbol{\beta}_k
  + \lambda_k u_i,\ \sigma_k^2\bigr),
```

and marginalising the Gaussian ``u_i`` keeps the pair Gaussian (a linear
combination of independent normals). The marginal moments are

```math
\mathbb{E}[y_{ki}] = \mathbf{x}_{ki}^\top\boldsymbol{\beta}_k,
\qquad
\operatorname{Var}(y_{ki}) = \lambda_k^2 + \sigma_k^2,
\qquad
\operatorname{Cov}(y_{1i}, y_{2i}) = \lambda_1\lambda_2,
```

so

```math
(y_{1i}, y_{2i}) \sim \mathcal{N}\!\left(
  \begin{pmatrix}\mathbf{x}_{1i}^\top\boldsymbol{\beta}_1 \\ \mathbf{x}_{2i}^\top\boldsymbol{\beta}_2\end{pmatrix},\
  \begin{pmatrix}\lambda_1^2 + \sigma_1^2 & \lambda_1\lambda_2 \\
                 \lambda_1\lambda_2 & \lambda_2^2 + \sigma_2^2\end{pmatrix}
\right).
```

This is exactly the bivariate Gaussian fitted by
`bf(mu1=…, mu2=…, sigma1=…, sigma2=…, rho12=…)`, whose residual correlation is

```math
\rho_{12} = \frac{\operatorname{Cov}(y_{1i}, y_{2i})}
                 {\sqrt{\operatorname{Var}(y_{1i})\operatorname{Var}(y_{2i})}}
          = \frac{\lambda_1\lambda_2}{\sqrt{(\lambda_1^2 + \sigma_1^2)(\lambda_2^2 + \sigma_2^2)}}.
```

Substituting the Gaussian link-residual ``v_k = \sigma_k^2`` (Section 4) into the
latent-scale correlation gives

```math
\rho = \frac{\lambda_1\lambda_2}{\sqrt{(\lambda_1^2 + v_1)(\lambda_2^2 + v_2)}}
     = \frac{\lambda_1\lambda_2}{\sqrt{(\lambda_1^2 + \sigma_1^2)(\lambda_2^2 + \sigma_2^2)}}
     = \rho_{12}.
```

The marginal model is the same as the `rho12` bivariate Gaussian model. A finite
Gauss–Hermite rule is still a numerical quadrature approximation here: its
polynomial-degree exactness does not make a Gaussian-density integrand exact.
Thus fitted likelihood values need numerical agreement checks at a stated
tolerance. The mathematical equivalence identifies the same ``\log L`` and
reported correlation: the loading split is the unidentified ridge, while
``\rho = \rho_{12}`` is the trustworthy cross-axis summary. This equivalence is
used as a correctness anchor for the engine, with numerical parity checked
separately.

## 8. Summary of the construction

| Object | Expression | Where |
|---|---|---|
| Linear predictor | ``\eta_{ki} = \mathbf{x}_{ki}^\top\boldsymbol{\beta}_k + \lambda_k u_i`` | §2 |
| Shared latent | ``u_i \sim \mathcal{N}(0,1)`` | §2 |
| Dispersion sub-model | ``\log d_{ki} = \mathbf{x}^\sigma_{ki}{}^{\!\top}\boldsymbol{\beta}^\sigma_k`` | §2 |
| Marginal likelihood | ``\prod_i \int p_{1i}\,p_{2i}\,\phi(u)\,\mathrm{d}u`` via ``K``-node Gauss–Hermite | §3 |
| Link-residual variance | ``v_k`` (Nakagawa & Schielzeth 2010, per family) | §4 |
| Latent correlation | ``\rho = \dfrac{\lambda_1\lambda_2}{\sqrt{(\lambda_1^2+v_1)(\lambda_2^2+v_2)}}`` | §5 |
| Sign fix | ``\lambda_1 = \exp(\ell\lambda_1) > 0`` | §5 |
| CIs | Fisher-``z`` Wald · profile likelihood · parametric bootstrap | §6 |
| Gaussian × Gaussian | ``\equiv`` `rho12` bivariate Gaussian, ``\rho = \rho_{12}`` | §7 |

## References

- Nakagawa, S., & Schielzeth, H. (2010). Repeatability for Gaussian and
  non-Gaussian data: a practical guide for biologists. *Biological Reviews*,
  85(4), 935–956. — distribution-specific link-scale variances ``v_k``.
- Nakagawa, S., Johnson, P. C. D., & Schielzeth, H. (2017). The coefficient of
  determination ``R^2`` and intra-class correlation coefficient from generalized
  linear mixed-effects models revisited and expanded. *Journal of the Royal
  Society Interface*, 14(134), 20170213. — the observation/link-scale variance
  decomposition used here.
- Skrondal, A., & Rabe-Hesketh, S. (2004). *Generalized Latent Variable
  Modeling: Multilevel, Longitudinal, and Structural Equation Models.* Chapman &
  Hall/CRC. — shared-latent (common-factor) construction for cross-response
  dependence.
- Niku, J., Hui, F. K. C., Taskinen, S., & Warton, D. I. (2019). gllvm: Fast
  analysis of multivariate abundance data with generalized linear latent variable
  models in R. *Methods in Ecology and Evolution*, 10(12), 2173–2182. — GLLVM
  latent-variable dependence and link-scale standardisation (the gllvmTMB
  `link_residual_per_trait` lineage this implementation mirrors).
- Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules.
  *Mathematics of Computation*, 23(106), 221–230. — the Gauss–Hermite
  nodes/weights.

## See also

- [Choosing response families](distribution-families.md) — the families that can
  appear on either axis.
- [Which scale are you modelling?](which-scale.md) — link vs response scale.
- [Laplace vs variational marginals](marginal-la-vs-va.md) — why a 1-D quadrature
  beats a Laplace match for the shape/dispersion parameters.
- [Cross-family models](../cross-family.md) — fitting examples and supported
  family pairs.
