# fit_derivatives.jl — documented gradient / Hessian accessors of a fitted
# model (#971, owner decision 8a) and the `control_info` report (#994).

"""
    gradient(fit::DrmFit) -> Vector{Float64}

Gradient of the fit's **objective** at the estimate, `∇ fit.nll(θ̂)`.

**Scale and order.** The parameters are the internal ones that `coef(fit)`
returns: in `coef(fit)` order, named by `coefnames(fit)`, on the scale `coef`
reports (for example `log σ` for a `sigma` block, link-scale coefficients for
the mean). `vcov(fit)` uses the same order and scale.

**Objective.** The objective is the negative log-likelihood the optimiser
minimised: the ML negative log-likelihood on ML fits, including the prior-weighted
one when `weights` were given; the negative REML criterion on REML fits; the
penalised objective on MAP fits. On random-effect routes it is the route's
marginal approximation (Laplace, adaptive quadrature or the negative ELBO).

At a stationary optimum, every entry is close to zero. Together with
[`hessian`](@ref) it gives the scaled Newton decrement `g' H⁻¹ g`, a stationarity
check that does not rely on the optimiser's own `converged` flag.

The gradient comes from the route's exact gradient callback when it has one,
otherwise from forward-mode automatic differentiation. If the objective does
not accept dual numbers, it comes from central finite differences. A fit with
no stored objective throws an `ArgumentError`.

# Example
```julia
fit = drm(bf(@formula(dist ~ 1 + speed), @formula(sigma ~ 1 + speed)), Gaussian(); data = cars)
g = gradient(fit)                 # length(coef(fit)), near zero at the optimum
H = hessian(fit)
g' * (H \\ g)                      # scaled Newton decrement
```
"""
function gradient(fit::DrmFit)
    nll = _fit_objective(fit, "gradient")
    θ = Vector{Float64}(fit.theta)
    if fit.nllgrad !== nothing
        g = zeros(length(θ))
        out = fit.nllgrad(g, θ)
        return Vector{Float64}(out === nothing ? g : out)
    end
    try
        return ForwardDiff.gradient(nll, θ)
    catch e
        e isa InterruptException && rethrow()
        return _fd_gradient(nll, θ)
    end
end

"""
    hessian(fit::DrmFit) -> Matrix{Float64}

Hessian of the fit's objective at the estimate, `∇² fit.nll(θ̂)`: the observed
information on the same parameter scale and in the same `coef(fit)` order as
[`gradient`](@ref) (see there for the objective on each route).

On the fixed-effect routes `vcov(fit)` is the guarded inverse of this matrix,
so `inv(hessian(fit)) ≈ vcov(fit)` there. Some random-effect and structured
routes build `vcov(fit)` another way, for example from the exact Laplace
information of the q=4 engine or from a profiled objective. On those routes the
two can differ by approximation error. The Hessian is recomputed on demand, so
it is also available for a fit made with `control = DRMControl(se = false)`
("point estimates first, standard errors later").

It is computed by forward-mode automatic differentiation. When the objective
does not accept dual numbers, central finite differences of the gradient are
used instead. A fit with no stored objective throws an `ArgumentError`.

# Example
```julia
fit = drm(f, Gaussian(); data = cars, control = DRMControl(se = false))
V = inv(hessian(fit))             # the covariance, computed later
sqrt.(diag(V))                    # standard errors
```
"""
function hessian(fit::DrmFit)
    nll = _fit_objective(fit, "hessian")
    θ = Vector{Float64}(fit.theta)
    H = try
        ForwardDiff.hessian(nll, θ)
    catch e
        e isa InterruptException && rethrow()
        gfun = fit.nllgrad === nothing ? (x -> _fd_gradient(nll, x)) :
                                         (x -> (g = zeros(length(x)); o = fit.nllgrad(g, x); o === nothing ? g : o))
        _fd_jacobian(gfun, θ)
    end
    return Matrix{Float64}((H + H') / 2)
end

function _fit_objective(fit::DrmFit, what::AbstractString)
    fit.nll === nothing && throw(ArgumentError(
        "$what(fit): this route did not store its objective (`fit.nll`), so the " *
        "$what cannot be evaluated"))
    return fit.nll
end

function _fd_gradient(f, θ::Vector{Float64})
    g = similar(θ)
    for j in eachindex(θ)
        h = cbrt(eps()) * max(abs(θ[j]), 1.0)
        tp = copy(θ); tp[j] += h
        tm = copy(θ); tm[j] -= h
        g[j] = (f(tp) - f(tm)) / (2h)
    end
    return g
end

function _fd_jacobian(gfun, θ::Vector{Float64})
    p = length(θ)
    J = zeros(p, p)
    for j in 1:p
        h = cbrt(eps()) * max(abs(θ[j]), 1.0)
        tp = copy(θ); tp[j] += h
        tm = copy(θ); tm[j] -= h
        J[:, j] = (gfun(tp) .- gfun(tm)) ./ (2h)
    end
    return J
end

"""
    control_info(fit::DrmFit) -> Union{DRMControlReport, Nothing}

What `drm(...; control, weights)` did for this fit (a [`DRMControlReport`](@ref)),
or `nothing` when neither keyword was given. Use it to see the log-likelihood
reached from each `multi_start` start (`start_logliks`), which start was kept
(`kept_start`), whether the starts agree (`starts_agree`), and whether
`se = false` skipped the Hessian or discarded it (`se`).

# Example
```julia
fit = drm(f, Gamma(); data = dat, control = DRMControl(multi_start = 5))
info = control_info(fit)
info.start_logliks, info.kept_start, info.starts_agree
```
"""
control_info(fit::DrmFit) = fit.control_info
