using DRModels, Test
import DelimitedFiles

# Wave-10 gap map, rank 4: three of the five non-converged cells were reporting failure
# although the likelihood and drmTMB 0.8.1 agree. Each cell below is the wave-8 fuzz cell
# (400 rows, 40 groups; data in fixtures/nonconv_triage, tree and coordinates shared with
# fixtures/gamma_sdz). R reference values: drmTMB 0.8.1 ML, library of the wave-8 campaign.
#
#   Gamma phylo `sd-z`  : Fminbox L-BFGS stopped with a zero step at |grad|/n = 1.01e-6, just
#                         outside the 1e-6 reporting limit, at R's log-likelihood to 1e-10.
#                         Newton polish in `_fit_gamma_structured_sd_laplace`.
#   Gamma `(1|g)+(0+x1|g)`: the inner Newton solve of `_fit_indep_re_laplace` stalled at the
#                         rounding floor, reported failure (1e18) at benign points and BFGS stopped
#                         9.25 log-likelihood units below R at a point with |grad| = 92.
#   Poisson coordinate spatial: the warm-start buffer of the inner mode was never reset after an
#                         extreme first line-search trial, so every later evaluation returned 1e18.

const _NCT_DIR = joinpath(@__DIR__, "fixtures", "nonconv_triage")
const _NCT_SHARED = joinpath(@__DIR__, "fixtures", "gamma_sdz")

function _nct_data(name)
    m, h = DelimitedFiles.readdlm(joinpath(_NCT_DIR, name), ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    return (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")),
            x3 = Float64.(col("x3")), z = Float64.(col("z")), g = String.(col("g")))
end

@testset "Gamma phylo sd-z q1: converged at R's log-likelihood" begin
    d = _nct_data("data_gamma_phylo.csv")
    tree = strip(read(joinpath(_NCT_SHARED, "tree.nwk"), String))
    fit = drm(bf(@formula(y ~ x1 + x2 + x3 + phylo(1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g, phylogenetic) ~ z)), DRModels.Gamma(); data = d, tree)
    @test loglikelihood(fit) ≈ -621.4519749318 atol = 1e-6
    @test fit.converged
    # the reported flag answers the documented question: mean |gradient| per observation <= 1e-6
    # (this objective is finite-difference only, as in the route: central differences, step 2e-4 (1 + |theta|))
    g = map(eachindex(fit.theta)) do j
        h = 2e-4 * (1 + abs(fit.theta[j])); e = zeros(length(fit.theta)); e[j] = h
        (fit.nll(fit.theta .+ e) - fit.nll(fit.theta .- e)) / (2h)
    end
    @test maximum(abs, g) / length(d.y) < 1e-6
end

@testset "Gamma (1 | g) + (0 + x1 | g): R's optimum, same objective" begin
    d = _nct_data("data_gamma_ordinary.csv")
    fit = drm(bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g)), @formula(sigma ~ 1)),
              DRModels.Gamma(); data = d)
    thetaR = [0.4059919, 0.3834418, 0.3696287, -0.2123565, -0.5302756, -0.6333515, -1.0811218]
    # one objective: R's estimate scores R's log-likelihood (the stop was never an objective gap)
    @test -fit.nll(thetaR) ≈ -546.3427321263 atol = 1e-5
    @test loglikelihood(fit) ≈ -546.3427321263 atol = 1e-6
    @test fit.converged
    @test fit.theta ≈ thetaR atol = 1e-3
    # the inner solve no longer depends on its warm start at a benign point between the old
    # stop and R's optimum (it returned 1e18 there from either start)
    thold = [0.5652739952744578, 0.4109701074138554, 0.3457095214788571, -0.20165674748465925,
             -0.6343119583189081, -0.8546042373729174, -1.1175563540577056]
    for t in 0.0:0.1:1.0
        @test fit.nll((1 - t) .* thold .+ t .* thetaR) < 1e17
    end
end

@testset "Poisson coordinate spatial: fits, converged, at drmTMB's fixed-range optimum" begin
    # Wave-10 spatial parity (docs/dev-log/2026-11-09-spatial-parity.md): the route no longer
    # estimates the range; it uses drmTMB's fixed kernel exp(-d / median positive d) + 1e-6 I
    # (5 parameters, drmTMB 0.8.1 logLik -675.6946875). The earlier 1e18-sentinel regression
    # (a stale warm start after an extreme first line-search trial) belonged to the removed
    # estimated-range marginal; the fixed-range route is the sparse relmat Laplace fit.
    d = _nct_data("data_poisson_spatial.csv")
    m = DelimitedFiles.readdlm(joinpath(_NCT_SHARED, "coords.csv"), ',', Any; skipstart = 1)
    coords = Dict(String(m[i, 1]) => [Float64(m[i, 2]), Float64(m[i, 3])] for i in 1:size(m, 1))
    fit = drm(bf(@formula(y ~ x1 + x2 + x3 + spatial(1 | g))), Poisson(); data = d, coords)
    @test fit.converged
    @test length(fit.theta) == 5
    @test loglikelihood(fit) ≈ -675.6946875075 atol = 1e-4
end
