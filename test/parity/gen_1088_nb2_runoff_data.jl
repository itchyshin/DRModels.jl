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
