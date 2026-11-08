# Input-validation fixes from the 2026-11-08 adversarial-data red team
# (docs/dev-log/parity-crosswalk-evidence-20261107/red-team/README.md). One testset per fix:
#   1. `sd(g) ~ …` with fewer than two groups is refused up front, naming the cause.
#   2. Binomial structured SD routes accept `ids = …` (it was an unsupported-keyword MethodError).
#   3. Binomial q1 spatial SD accepts a label => coordinate Dict (it was an ErrorException).
#   4. Two refusal messages now say what to try (extra tree tips, singular relatedness matrix).
#   5. Binomial `(1 + x | g)` is refused only when the slope predictor is constant in EVERY group
#      (drmTMB's rule); one singleton group among varying ones used to be refused.
# Data are generated with StableRNG(20261108); no committed fixture is needed.
using DRModels, Test, LinearAlgebra, StableRNGs
import Distributions

function _rt_data(; G = 16, m = 10, seed = 20261108, kind = :binomial)
    rng = StableRNG(seed)
    ids = ["s$(lpad(i, 2, '0'))" for i in 1:G]
    g = repeat(ids, inner = m)
    x = randn(rng, G * m)
    z = repeat(repeat([-1.0, 1.0], G ÷ 2), inner = m)
    # shared structured random effect (AR(1) kernel) so the animal / spatial fits are well posed
    R0 = 0.5 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    D = Diagonal(collect(range(0.7, 1.4; length = G)))      # non-Toeplitz: a reversed matrix is a different model
    R = Matrix(D * R0 * D)
    u = cholesky(Symmetric(R)).L * randn(rng, G)
    sdg = exp.(-0.3 .+ 0.3 .* repeat([-1.0, 1.0], G ÷ 2))
    eta = 0.2 .+ 0.4 .* x .+ repeat(sdg .* u, inner = m)
    y = kind === :binomial ?
        Float64.(rand(rng, G * m) .< 1 ./ (1 .+ exp.(-eta))) :
        Float64[rand(rng, Distributions.Poisson(exp(v))) for v in eta]
    return (; data = (; y, x, z, g), ids, R)
end

function _rt_throws_arg(f, needle)
    err = try f(); nothing catch e; e end
    err isa ArgumentError || return false
    return occursin(needle, err.msg)
end

@testset "red team: sd() with fewer than two groups is refused up front" begin
    fx = _rt_data(kind = :poisson)
    one = merge(fx.data, (; g = fill("s01", length(fx.data.y)), z = fill(1.0, length(fx.data.y))))
    f = bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z))
    for fam in (Gaussian(), Poisson(), Binomial())
        d = fam isa Binomial ? merge(one, (; y = Float64.(one.y .> 0))) : one
        @test _rt_throws_arg(() -> drm(f, fam; data = d), "At least two groups are needed")
    end
    # the same guard covers a structured route and names the grouping variable
    fa = bf(@formula(y ~ x + animal(1 | g)), @formula(sd(g, animal) ~ z))
    @test _rt_throws_arg(() -> drm(fa, Gaussian(); data = one, A = ones(1, 1)), "`g` has only one level")
    # negative control: two groups pass the guard (the fit itself may or may not be sensible)
    two = (; y = fx.data.y[1:20], x = fx.data.x[1:20], z = fx.data.z[1:20], g = fx.data.g[1:20])
    err = try drm(bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z)), Gaussian(); data = two); nothing catch e; e end
    @test !(err isa ArgumentError && occursin("At least two groups are needed", err.msg))
end

@testset "red team: Binomial animal SD route accepts ids = …" begin
    fx = _rt_data()
    f = bf(@formula(y ~ x + animal(1 | g)), @formula(sd(g, animal) ~ z))
    base = drm(f, Binomial(); data = fx.data, A = fx.R, se = false)
    p = reverse(1:length(fx.ids))                      # a permuted, labelled matrix
    labelled = drm(f, Binomial(); data = fx.data, A = fx.R[p, p], ids = fx.ids[p], se = false)
    @test labelled.converged
    @test loglikelihood(labelled) ≈ loglikelihood(base) rtol = 1e-8
    # without `ids` the permuted matrix is read by position: a different (wrong) model
    @test !isapprox(loglikelihood(drm(f, Binomial(); data = fx.data, A = fx.R[p, p], se = false)),
                    loglikelihood(base); rtol = 1e-6)
    @test _rt_throws_arg(() -> drm(f, Binomial(); data = fx.data, Ainv = inv(fx.R), ids = fx.ids, se = false),
                         "labels the rows of `A`")
    plain = bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z))
    @test _rt_throws_arg(() -> drm(plain, Binomial(); data = fx.data, ids = fx.ids, se = false),
                         "neither was supplied")
end

@testset "red team: Binomial q1 spatial SD route accepts a label => coordinate Dict" begin
    fx = _rt_data()
    G = length(fx.ids)
    xy = [(Float64(i % 4), Float64(i ÷ 4)) for i in 0:G-1]
    mat = reduce(vcat, [[a b] for (a, b) in xy])
    f = bf(@formula(y ~ x + spatial(1 | g)), @formula(sd(g, spatial) ~ z))
    base = drm(f, Binomial(); data = fx.data, coords = mat, se = false)       # first-seen order = ids order
    p = reverse(1:G)
    dict = Dict(fx.ids[i] => [xy[i][1], xy[i][2]] for i in p)                # insertion order scrambled
    byid = drm(f, Binomial(); data = fx.data, coords = dict, se = false)
    @test byid.converged
    @test loglikelihood(byid) ≈ loglikelihood(base) rtol = 1e-8
    viaids = drm(f, Binomial(); data = fx.data, coords = mat[p, :], ids = fx.ids[p], se = false)
    @test loglikelihood(viaids) ≈ loglikelihood(base) rtol = 1e-8
    missing_site = Dict(k => v for (k, v) in dict if k != "s03")
    @test _rt_throws_arg(() -> drm(f, Binomial(); data = fx.data, coords = missing_site, se = false),
                         "no row in `coords`")
end

@testset "red team: refusal messages say what to try" begin
    fx = _rt_data(kind = :poisson)
    G = length(fx.ids)
    # (a) a tree with tips that have no data rows (R accepts a larger tree)
    all_ids = vcat(fx.ids, ["s17", "s18"])
    nw(ix) = length(ix) == 1 ? "$(all_ids[ix[1]]):1.0" :
        "(" * nw(ix[1:end÷2]) * "," * nw(ix[end÷2+1:end]) * "):0.5"
    tree = nw(collect(1:length(all_ids))) * ";"
    fp = bf(@formula(y ~ x + phylo(1 | g)), @formula(sd(g, phylogenetic) ~ z))
    @test _rt_throws_arg(() -> drm(fp, Gaussian(); data = fx.data, tree), "ape::keep.tip")
    # (b) exactly duplicated relatives make A singular; the Poisson animal SD route used to call it `relmat`
    A = Matrix(fx.R); A[2, :] = A[1, :]; A[:, 2] = A[:, 1]; A[2, 2] = A[1, 1]
    fa = bf(@formula(y ~ x + animal(1 | g)), @formula(sd(g, animal) ~ z))
    err = try drm(fa, Poisson(); data = fx.data, A, ids = fx.ids, se = false); nothing catch e; e end
    msg = err === nothing ? "" : sprint(showerror, err)
    # which of the two checks fires depends on how close to singular the rounded factor is; both must explain
    @test err isa Union{ArgumentError,ErrorException}
    @test occursin("not positive definite", msg) && occursin("duplicated", msg) && !occursin("relmat(1 | g)", msg)
end

@testset "red team: Binomial (1 + x | g) slope check refuses only when x is constant in every group" begin
    rng = StableRNG(20261108)
    G, m = 12, 8
    g = repeat(["s$(lpad(i, 2, '0'))" for i in 1:G], inner = m)
    x = randn(rng, G * m)
    z = repeat(repeat([-1.0, 1.0], G ÷ 2), inner = m)
    u = randn(rng, G)
    y = Float64.(rand(rng, G * m) .< 1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* x .+ 0.7 .* repeat(u, inner = m)))))
    f = bf(@formula(y ~ x + (1 + x | g)),
           @formula(sd(g, coef(intercept)) ~ z), @formula(sd(g, coef(x)) ~ z))
    keep = trues(G * m); keep[(m + 2):(2m)] .= false                 # group s02 keeps one row: x trivially constant there
    one_singleton = (; y = y[keep], x = x[keep], z = z[keep], g = g[keep])
    fit = drm(f, Binomial(); data = one_singleton, se = false)
    @test isfinite(loglikelihood(fit))
    const_x = (; y, x = repeat(randn(rng, G), inner = m), z, g)       # x constant in every group
    err = try drm(f, Binomial(); data = const_x, se = false); nothing catch e; e end
    @test err isa ErrorException && occursin("constant within every level", err.msg)
end
