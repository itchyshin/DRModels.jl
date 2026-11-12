# Writes d<seed>.csv: the three simulated data sets of the extra-starts regression test (seeds 80, 132 and
# 135 of the sweep that found the lower local mode). Not run by the tests: the CSV files are the stored
# truth. The stream reproduces them on Julia 1.13.3; Julia 1.10 draws different numbers from the same
# seeds. Same generator as ../biv_re_domainfix/make_data.jl. Usage: julia make_data.jl <outdir> [seed ...]
# Unbalanced groups, group SDs from {0.03, 0.1, 0.3, 0.7}, group correlation from {-0.9, ..., 0.9}.
using Random, LinearAlgebra
function params(s)
    rng = MersenneTwister(50000 + s)
    n = rand(rng, [40, 60, 100, 160, 240]); G = rand(rng, [4, 6, 10, 16, 24])
    sd1 = rand(rng, [0.03, 0.1, 0.3, 0.7]); sd2 = rand(rng, [0.03, 0.1, 0.3, 0.7])
    ru = rand(rng, [-0.9, -0.5, 0.0, 0.2, 0.5, 0.9]); rr = rand(rng, [-0.5, 0.0, 0.4])
    (; n, G, sd1, sd2, ru, rr)
end
function sim(s)
    p = params(s); rng = MersenneTwister(60000 + s)
    n, G = p.n, p.G
    g = vcat(1:G, [min(G, 1 + floor(Int, G * rand(rng)^2)) for _ in 1:(n-G)])
    L = cholesky([p.sd1^2 p.ru*p.sd1*p.sd2; p.ru*p.sd1*p.sd2 p.sd2^2]).L
    U = L * randn(rng, 2, G)
    x1 = randn(rng, n); x2 = randn(rng, n); e1 = randn(rng, n); e2 = randn(rng, n)
    y1 = 0.3 .+ 0.5 .* x1 .+ U[1, g] .+ 0.7 .* e1
    y2 = -0.2 .+ 0.4 .* x2 .+ U[2, g] .+ 0.9 .* (p.rr .* e1 .+ sqrt(1 - p.rr^2) .* e2)
    (; y1, y2, x1, x2, g = ["h$(lpad(k, 2, '0'))" for k in g])
end
out = ARGS[1]; mkpath(out)
for s in (length(ARGS) > 1 ? parse.(Int, ARGS[2:end]) : [80, 132, 135])
    d = sim(s)
    open(joinpath(out, "d$(s).csv"), "w") do io
        println(io, "id,g,x1,x2,y1,y2")
        for i in eachindex(d.y1); println(io, i, ",", d.g[i], ",", d.x1[i], ",", d.x2[i], ",", d.y1[i], ",", d.y2[i]); end
    end
end
