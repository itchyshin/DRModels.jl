# gen_1088_nb2_runoff_data.jl -- the three #1088 datasets (NB2 crossed, heavy overdispersion).
#
#   julia --project=. test/parity/gen_1088_nb2_runoff_data.jl
#
# Writes test/fixtures/j3_se/nb2_runoff_s<seed>_n<n>/data.csv for the issue's three sets
# (MersenneTwister(7) with n = 1500 and n = 800; MersenneTwister(8) with n = 1500). The body
# is the pasteable twin of the original generator posted in #1088 (Code Reviewer, #1075
# review): G = 25, H = 20, RE SD 0.3, eta = 5 + 0.3x, NB size r = 0.0022. The data are written
# once and committed, so the tests do not depend on the RNG stream of a future Distributions
# release. The drmTMB reference for each set is then produced from these CSVs by
# test/parity/gen_1088_nb2_runoff_reference.R.
#
# A fourth set, nb2_disp3to4_s1_n5000, is a real fit whose log-dispersion lands in (3, 4)
# with interior RE SDs (size 0.0015, RE SD 1.2, 6 x 6 levels), for the gradient-path check
# next to the theta_sigma = 3.5 point evaluation in test_j3a_gradient_hessian_vcov.jl. Its x
# is written to 4 decimals so the CSV stays small; both engines fit the CSV as written.
using Random, Distributions

const OUT = joinpath(@__DIR__, "..", "fixtures", "j3_se")

function gen_1088(seed::Int, n::Int)
    rng = MersenneTwister(seed); G, H = 25, 20
    g = ["g$(rand(rng, 1:G))" for _ in 1:n]
    h = ["h$(rand(rng, 1:H))" for _ in 1:n]
    bg = Dict("g$j" => 0.3randn(rng) for j in 1:G)
    bh = Dict("h$j" => 0.3randn(rng) for j in 1:H)
    x = randn(rng, n); r = 0.0022
    y = [Float64(rand(rng, NegativeBinomial(r, r / (r + exp(5 + 0.3x[i] + bg[g[i]] + bh[h[i]])))))
         for i in 1:n]
    return (; y, x, g, h)
end

function gen_disp3to4(seed::Int, n::Int; G = 6, H = 6, sd = 1.2, b0 = 3.0, b1 = 0.4, r = 0.0015)
    rng = MersenneTwister(seed)
    g = ["g$(rand(rng, 1:G))" for _ in 1:n]
    h = ["h$(rand(rng, 1:H))" for _ in 1:n]
    bg = Dict("g$j" => sd * randn(rng) for j in 1:G)
    bh = Dict("h$j" => sd * randn(rng) for j in 1:H)
    x = round.(randn(rng, n); digits = 4)
    y = [Float64(rand(rng, NegativeBinomial(r, r / (r + exp(b0 + b1 * x[i] + bg[g[i]] + bh[h[i]])))))
         for i in 1:n]
    return (; y, x, g, h)
end

function write_set(route, d)
    dir = joinpath(OUT, route)
    mkpath(dir)
    open(joinpath(dir, "data.csv"), "w") do io
        println(io, "y,x,g,h")
        for i in eachindex(d.y)
            # Julia prints the shortest decimal that round-trips the Float64 exactly
            println(io, round(Int, d.y[i]), ",", d.x[i], ",", d.g[i], ",", d.h[i])
        end
    end
    println("wrote ", dir, "  n=", length(d.y), "  nonzero y = ", count(>(0), d.y), "  max y = ", maximum(d.y))
end

for (seed, n) in ((7, 1500), (7, 800), (8, 1500))
    write_set("nb2_runoff_s$(seed)_n$(n)", gen_1088(seed, n))
end
write_set("nb2_disp3to4_s1_n5000", gen_disp3to4(1, 5000))

# Gamma crossed run-off (same engine as NB2; recover_runoff=true is the default).
# Shape 0.002, G=15, H=12, n=200, MersenneTwister(7), η = 2 + 0.3x, RE SD 0.3.
# y is continuous, so this writer keeps the shortest round-trip Float64.
function gen_gamma_runoff(seed::Int, n::Int; G=15, H=12, shape=0.002, b0=2.0, b1=0.3, sd=0.3)
    rng = MersenneTwister(seed)
    g = ["g$(rand(rng, 1:G))" for _ in 1:n]
    h = ["h$(rand(rng, 1:H))" for _ in 1:n]
    bg = Dict("g$j" => sd * randn(rng) for j in 1:G)
    bh = Dict("h$j" => sd * randn(rng) for j in 1:H)
    x = randn(rng, n)
    μ = [exp(b0 + b1 * x[i] + bg[g[i]] + bh[h[i]]) for i in 1:n]
    y = max.([rand(rng, Gamma(shape, μ[i] / shape)) for i in 1:n], 1e-12)
    return (; y, x, g, h)
end
let d = gen_gamma_runoff(7, 200)
    dir = joinpath(OUT, "gamma_runoff_s7_n200")
    mkpath(dir)
    open(joinpath(dir, "data.csv"), "w") do io
        println(io, "y,x,g,h")
        for i in eachindex(d.y)
            println(io, d.y[i], ",", d.x[i], ",", d.g[i], ",", d.h[i])
        end
    end
    println("wrote ", dir, "  n=", length(d.y), "  ymax = ", maximum(d.y))
end

# Beta crossed run-off (same engine; recover_runoff=true is the default). Precision 0.01, so
# the draws pile up at the 0 / 1 edges (clamped to [1e-100, 1 - 1e-12]). G=15, H=12, n=300,
# MersenneTwister(3), logit μ = 0.2 + 0.3x, RE SDs 0.5 / 0.4. Written with Julia 1.12.6: the
# MersenneTwister draws differ on Julia 1.10, and on the 1.10 draws the start objective is
# already the failed-inner-solve sentinel, so there is no run-off to recover.
function gen_beta_runoff(seed::Int, n::Int; G=15, H=12, φ=0.01)
    rng = MersenneTwister(seed)
    gi = rand(rng, 1:G, n); hi = rand(rng, 1:H, n); x = randn(rng, n)
    bg = 0.5 .* randn(rng, G); bh = 0.4 .* randn(rng, H)
    μ = 1 ./ (1 .+ exp.(-(0.2 .+ 0.3 .* x .+ bg[gi] .+ bh[hi])))
    y = [clamp(rand(rng, Beta(μ[i] * φ, (1 - μ[i]) * φ)), 1e-100, 1 - 1e-12) for i in 1:n]
    return (; y, x, g = ["g$i" for i in gi], h = ["h$i" for i in hi])
end
let d = gen_beta_runoff(3, 300)
    dir = joinpath(OUT, "beta_runoff_s3_n300")
    mkpath(dir)
    open(joinpath(dir, "data.csv"), "w") do io
        println(io, "y,x,g,h")
        for i in eachindex(d.y)
            println(io, d.y[i], ",", d.x[i], ",", d.g[i], ",", d.h[i])
        end
    end
    println("wrote ", dir, "  n=", length(d.y), "  at 0 / 1 edges = ",
            count(==(1e-100), d.y), " / ", count(==(1 - 1e-12), d.y))
end
