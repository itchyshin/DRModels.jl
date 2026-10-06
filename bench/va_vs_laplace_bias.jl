# bench/va_vs_laplace_bias.jl — #136e public-path Gamma RI smoke (LA vs VA).
#
# ADEMP cell matches test/test_va_frontend_families.jl Gamma fixture:
#   α=4, G=40, per=10, β=(0.4, 0.5), σb=0.5.
# Estimand: shape α = exp(-2 · coef(fit, :sigma)[1]).
# Public API only: drm(...; marginal=:LA|:VA). ELBO ≠ logLik — do not compare as IC.
#
# Run (repo root):
#   julia --project=. --startup-file=no bench/va_vs_laplace_bias.jl
#   julia --project=. --startup-file=no bench/va_vs_laplace_bias.jl 3
#
# Grep: VA-vs-LA

using DRModels
using Random
using Printf
import Distributions

const α_TRUE = 4.0
const G = 40
const PER = 10
const β0, β1 = 0.4, 0.5
const σb_TRUE = 0.5
const SEEDS = (202608082, 202608083, 202608084)  # first seed = frontend Gamma test

αhat(fit) = exp(-2 * coef(fit, :sigma)[1])

function simulate(rng)
    n = G * PER
    g = repeat(1:G, inner = PER)
    x = randn(rng, n)
    bg = σb_TRUE .* randn(rng, G)
    μ = exp.(β0 .+ β1 .* x .+ bg[g])
    y = Float64.([rand(rng, Distributions.Gamma(α_TRUE, μi / α_TRUE)) for μi in μ])
    return (; y, x, g)
end

function one_rep(seed)
    rng = MersenneTwister(seed)
    data = simulate(rng)
    f = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))
    t_la = @elapsed fit_la = drm(f, Gamma(); data = data, marginal = :LA)
    t_va = @elapsed fit_va = drm(f, Gamma(); data = data, marginal = :VA)
    α_la = αhat(fit_la)
    α_va = αhat(fit_va)
    σb_la = re_sd(fit_la)[:g]
    σb_va = re_sd(fit_va)[:g]
    @info "VA-vs-LA [Gamma-RI]" seed=seed α_true=α_TRUE α_LA=α_la α_VA=α_va bias_LA=(α_la - α_TRUE) bias_VA=(α_va - α_TRUE) relbias_LA=((α_la - α_TRUE) / α_TRUE) relbias_VA=((α_va - α_TRUE) / α_TRUE) σb_true=σb_TRUE σb_LA=σb_la σb_VA=σb_va β0_LA=coef(fit_la, :mu)[1] β0_VA=coef(fit_va, :mu)[1] β1_LA=coef(fit_la, :mu)[2] β1_VA=coef(fit_va, :mu)[2] obj_LA=loglik(fit_la) obj_VA=loglik(fit_va) t_LA=t_la t_VA=t_va ratio=t_va / t_la conv_LA=is_converged(fit_la) conv_VA=is_converged(fit_va) marginal_LA=fit_la.marginal marginal_VA=fit_va.marginal
    return (; seed, α_la, α_va, t_la, t_va, conv_la=is_converged(fit_la),
            conv_va=is_converged(fit_va), finite=isfinite(α_la) && isfinite(α_va))
end

nrep = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
nrep = clamp(nrep, 1, length(SEEDS))
println("=== #136e public Gamma RI  LA vs VA  nrep=$(nrep)  α_true=$(α_TRUE) ===")
rows = [one_rep(SEEDS[i]) for i in 1:nrep]
ok = all(r -> r.finite && r.conv_la && r.conv_va, rows)
println(ok ? "SMOKE_OK" : "SMOKE_FAIL")
for r in rows
    @printf "seed=%d  α_LA=%.4f  α_VA=%.4f  Δα=%.4f  t_LA=%.3fs  t_VA=%.3fs  ratio=%.2f\n" r.seed r.α_la r.α_va (r.α_va - r.α_la) r.t_la r.t_va (r.t_va / r.t_la)
end
