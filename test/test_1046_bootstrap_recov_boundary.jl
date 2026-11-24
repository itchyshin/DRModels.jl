# #1046: a (1 + x | g) bootstrap whose slope SD collapses must be flagged.
# `:recov` / `:phylocov` store Cholesky columns, so the boundary check unpacks
# each draw into intercept SD, slope SD and correlation before applying the
# usual cutoffs.
module Test1046BootstrapRecovBoundary

using DRModels
using Test
using Random

@testset "#1046 (1 + x | g) slope-SD collapse -> bootstrap_at_boundary" begin
    rng = MersenneTwister(1046)
    G, m = 20, 8
    n = G * m
    g = repeat(1:G, inner = m)
    x = randn(rng, n)
    a = 0.6 .* randn(rng, G)
    e = 0.5 .* randn(rng, n)
    for k in 1:G           # remove each group's own LS slope: true slope SD 0
        i = findall(==(k), g)
        X = hcat(ones(m), x[i])
        e[i] .-= X * (X \ e[i])
    end
    y = 0.2 .+ 0.4 .* x .+ a[g] .+ e
    data = (; y, x, g)
    form = bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1))
    fit0 = drm(form, Gaussian(); data)
    refit = d -> drm(form, Gaussian(); data = d)
    res = DRModels._bootstrap_result(fit0, form, data, 24, 0.95, MersenneTwister(1046),
                                     false, refit; failures = :skip)
    @test res.used >= 20
    @test res.status == "bootstrap_at_boundary"
    @test any(n -> startswith(n, "recov:") && endswith(n, ":sd_slope"), res.boundary_params)
end

@testset "#1046 Cholesky unpacking (unit)" begin
    r = 1:3
    nms = ["g:L11", "g:L22", "g:L21"]
    # slope SD ~ 0 and |rho| ~ 1 in every draw
    d = repeat([0.0 -30.0 1e-6], 25)
    out = DRModels._bootstrap_cholesky_boundary_names(:recov, r, nms, d)
    @test "recov:g:sd_slope" in out
    @test "recov:g:cor" in out
    @test !("recov:g:sd_intercept" in out)
    # interior draws are not flagged
    ok = repeat([0.0 0.0 0.2], 25)
    @test isempty(DRModels._bootstrap_cholesky_boundary_names(:recov, r, nms, ok))
    # phylocov order [L11, L21, L22] is resolved by name
    pn = ["Sigma_a:L11", "Sigma_a:L21", "Sigma_a:L22"]
    pd = repeat([0.0 1e-6 -30.0], 25)
    @test "phylocov:Sigma_a:sd_slope" in
          DRModels._bootstrap_cholesky_boundary_names(:phylocov, r, pn, pd)
    # other block shapes are skipped
    @test isempty(DRModels._bootstrap_cholesky_boundary_names(
        :recov, 1:1, ["g:atanh_rho"], repeat([5.0;;], 25)))
end

end
