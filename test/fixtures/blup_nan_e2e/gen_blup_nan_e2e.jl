# Generator for the committed fixtures in this directory (run once; the CSVs are the artefact).
#   julia --project=test test/fixtures/blup_nan_e2e/gen_blup_nan_e2e.jl
# Tiny Gaussian data: y ~ x + relmat(1 | g1) + animal(1 | g2), 6 levels each, n = 36.
#   K_pd.csv    : valid correlation matrix (0.5 off-diagonal)  -> V positive definite at every theta
#   K_indef.csv : symmetric, unit diagonal, off-diagonal 0.9 except two entries at -0.9; indefinite
#                 (see the eigenvalue check in the test), so V = s2 I + s1 Z K Z' + s2' Z A Z' is
#                 NOT positive definite at the default start.
using StableRNGs, DelimitedFiles, LinearAlgebra
rng = StableRNG(20261109)
G = 6; n = 36
g1 = repeat(1:G, inner = 6); g2 = repeat(1:G, outer = 6)
x = round.(randn(rng, n); digits = 4)
y = round.(0.5 .+ 0.3 .* x .+ randn(rng, n); digits = 4)
dir = @__DIR__
open(joinpath(dir, "data.csv"), "w") do io
    println(io, "y,x,g1,g2")
    for i in 1:n
        println(io, y[i], ",", x[i], ",", g1[i], ",", g2[i])
    end
end
Kpd = fill(0.5, G, G) + 0.5I
Kind = fill(0.9, G, G); Kind[diagind(Kind)] .= 1.0
Kind[1, 2] = Kind[2, 1] = -0.9; Kind[3, 4] = Kind[4, 3] = -0.9
writedlm(joinpath(dir, "K_pd.csv"), Kpd, ',')
writedlm(joinpath(dir, "K_indef.csv"), Kind, ',')
@show eigvals(Symmetric(Kind))
