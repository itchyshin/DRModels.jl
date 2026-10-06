# test_twin_gap_569.jl — #569: `drm_bridge` exports route-aware convergence
# diagnostics (`bridge_diagnostics`, src/introspection.jl) so the drmTMB R
# side (`check_drm.drmTMB_julia`, R/julia-diagnostics.R) can compare an
# `engine = "julia"` fit against TMB's own convergence checks. Covers three
# routes with different diagnostic shapes: a Gaussian closed-form fixed-effect
# fit (an `nllgrad` and an achieved iteration count), a non-Gaussian ordinary
# random-intercept Laplace route (an `nllgrad` but a route-specific
# "not-a-single-outer-optimiser" iteration story is NOT the case here — it IS
# wired per `niterations`'s own docstring), and a crossed-random-intercept
# Laplace route (BetaBinomial `(1 | g) + (1 | h)`, whose sparse engine records
# neither an achieved iteration count nor a stored gradient callback).

using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "bridge_diagnostics / drm_bridge diagnostics payload (#569)" begin
    # --- Gaussian closed-form fixed-effect route ------------------------------
    Random.seed!(20260927)
    n = 80
    x = collect(range(-1, 1; length = n))
    y = 0.3 .+ 0.8 .* x .+ 0.25 .* randn(n)
    data = (; y = y, x = x)

    native = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = data)
    d = bridge_diagnostics(native)

    @test d.route isa String
    @test d.integrator === native.marginal
    @test d.converged === native.converged
    @test d.iterations isa Int && d.iterations >= 0   # this route IS wired (niterations docstring)
    @test d.optimizer == "Optim.LBFGS"
    @test d.grad_source isa Symbol
    @test !ismissing(d.max_abs_grad)
    @test d.max_abs_grad isa Float64 && d.max_abs_grad <= 1e-2
    @test d.vcov_complete isa Bool
    if d.vcov_complete
        @test d.vcov_posdef isa Bool
        @test d.min_eigval isa Float64
        @test d.cond isa Float64
    end
    @test d.penalized_map == false
    @test d.boundary isa Vector{Int} && isempty(d.boundary)

    bridged = drm_bridge(; formula = "y ~ x; sigma ~ 1", family = "gaussian", data = data)
    @test haskey(bridged, "diagnostics")
    bd = bridged["diagnostics"]
    @test bd["route"] == d.route
    @test bd["integrator"] == String(d.integrator)
    @test bd["optimizer"] == d.optimizer
    @test bd["converged"] == d.converged
    @test bd["iterations"] == d.iterations
    @test bd["max_abs_grad"] ≈ d.max_abs_grad
    @test bd["grad_source"] == String(d.grad_source)
    @test bd["vcov_complete"] == d.vcov_complete
    @test bd["penalized_map"] == d.penalized_map
    @test bd["boundary"] == d.boundary
    # `grad_source` is also echoed at the top level: R/julia-diagnostics.R's
    # `drm_julia_gradient_source()` reads `object$bridge[["grad_source"]]` off
    # the RAW (un-nested) payload, not off a nested "diagnostics" key.
    @test bridged["grad_source"] == String(d.grad_source)

    # --- non-Gaussian ordinary random-intercept Laplace route -----------------
    G = 25
    m = 12
    N = G * m
    grp = repeat(1:G, inner = m)
    xri = randn(N)
    b = 0.3 .* randn(G)
    b .-= sum(b) / G
    eta = 0.1 .+ 0.4 .* xri .+ b[grp]
    mu_ri = exp.(eta)
    yri = Float64.(rand.(Ref(Random.default_rng()), Distributions.Poisson.(mu_ri)))
    ridata = (; y = yri, x = xri, g = Symbol.("g", grp))

    ri_fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), DRModels.Poisson();
                 data = ridata, marginal = :Laplace)
    d_ri = bridge_diagnostics(ri_fit)
    @test d_ri.integrator === :Laplace
    @test d_ri.route isa String
    @test d_ri.converged === ri_fit.converged
    @test d_ri.grad_source isa Symbol
    # Route-honesty: don't assert a specific iteration/gradient story for this
    # route beyond what `check_drm`/`niterations` themselves report — only that
    # the diagnostic record is self-consistent (missing pairs with missing).
    @test (d_ri.iterations isa Int && d_ri.iterations >= 0) || ismissing(d_ri.iterations)
    @test ismissing(d_ri.iterations) == ismissing(d_ri.optimizer)
    @test ismissing(d_ri.max_abs_grad) == (d_ri.grad_source in (:none, :unavailable))

    ri_bridged = drm_bridge(; formula = "y ~ x + (1 | g); sigma ~ 1", family = "poisson",
                            data = ridata, options = Dict(:marginal => "Laplace"))
    bd_ri = ri_bridged["diagnostics"]
    @test bd_ri["integrator"] == "Laplace"
    @test ismissing(bd_ri["iterations"]) == ismissing(d_ri.iterations)
    @test ismissing(bd_ri["max_abs_grad"]) == ismissing(d_ri.max_abs_grad)

    # --- crossed random-intercept Laplace route (BetaBinomial (1|g)+(1|h)) ----
    Gc = 18
    Hc = 15
    Nc = 900
    rng = MersenneTwister(20260927)
    gids = [rand(rng, 1:Gc) for _ in 1:Nc]
    hids = [rand(rng, 1:Hc) for _ in 1:Nc]
    gsym = [Symbol("g", j) for j in gids]
    hsym = [Symbol("h", j) for j in hids]
    xc = randn(rng, Nc)
    bg = 0.3 .* randn(rng, Gc); bg .-= sum(bg) / Gc
    bh = 0.2 .* randn(rng, Hc); bh .-= sum(bh) / Hc
    eta_c = [0.1 + 0.3 * xc[i] + bg[gids[i]] + bh[hids[i]] for i in 1:Nc]
    mu_c = 1 ./ (1 .+ exp.(-eta_c))
    precision = 15.0
    ntr = fill(6, Nc)
    successes = Float64.([rand(rng, Distributions.BetaBinomial(ntr[i], mu_c[i] * precision,
                                                                (1 - mu_c[i]) * precision)) for i in 1:Nc])
    failures = Float64.(ntr) .- successes
    crdata = (; successes = successes, failures = failures, x = xc, g = gsym, h = hsym)

    cr_fit = drm(bf(@formula(cbind(successes, failures) ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
                 DRModels.BetaBinomial(); data = crdata)
    d_cr = bridge_diagnostics(cr_fit)
    @test d_cr.converged === cr_fit.converged
    @test d_cr.integrator === cr_fit.marginal
    # The crossed sparse-Laplace engine has no single outer `Optim` call to
    # attribute an iteration count to (niterations's own docstring): honestly
    # `missing`, never a fabricated 0.
    @test ismissing(d_cr.iterations)
    @test ismissing(d_cr.optimizer)
    @test d_cr.grad_source isa Symbol
    @test ismissing(d_cr.max_abs_grad) == (d_cr.grad_source in (:none, :unavailable))
    @test d_cr.boundary isa Vector{Int}

    cr_bridged = drm_bridge(; formula = Dict(:mu => "cbind(successes, failures) ~ x + (1 | g) + (1 | h)",
                                             :sigma => "sigma ~ 1"),
                            family = "betabinomial", data = crdata)
    bd_cr = cr_bridged["diagnostics"]
    @test ismissing(bd_cr["iterations"])
    @test ismissing(bd_cr["optimizer"])
    @test bd_cr["converged"] == d_cr.converged
    @test cr_bridged["grad_source"] == String(d_cr.grad_source)
end
