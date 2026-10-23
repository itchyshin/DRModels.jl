# test_phylo_slope_two_sd.jl — the Gaussian two-SD phylogenetic random
# intercept + slope, `phylo(1 + x | species)` on `mu` (#620).
#
# THE MODEL (drmTMB's, `src/drmTMB.cpp` `has_phylo_mu` q = 2, both dpars = mu):
#     yᵢ = xᵢᵀβ + a_{s(i)} + b_{s(i)} xᵢ + εᵢ
#     a ~ N(0, σₐ² C),  b ~ N(0, σ_b² C),  a ⊥ b (NO intercept–slope correlation),
#     ε ~ N(0, σ² I),  C = tree correlation (tip covariance / height).
# The marginal is exactly Gaussian, V = σ² I + σₐ² Z C Zᵀ + σ_b² Zₓ C Zₓᵀ.
#
# FIXTURE (committed as literals below, so the drmTMB same-target numbers pinned
# here refer to EXACTLY this data): 16-tip balanced tree, every branch 0.25
# (4 levels → height 1, so C is already on the correlation scale), 8 rows per
# tip, x the grid range(-1, 1, 8) per species. Generated once with
# `Random.Xoshiro(20260905)`: β = (0.4, −0.25), σₐ = 0.55, σ_b = 0.32, σ = 0.22.
#
# SAME-TARGET ORACLE: drmTMB origin/main 67703f541 (native TMB, engine = "tmb")
# fitted on this fixture through `ape::stree(16, "balanced")` with all edge
# lengths 0.25 (same C — spot-checked C[1,2]=0.75, C[1,3]=0.5, C[1,5]=0.25,
# C[1,9]=0 on both sides), `bf(y ~ x + phylo(1 + x | species, tree = tree),
# sigma ~ 1)`, `opt$convergence == 0`:
#     logLik = -19.586599200945
#     sdpars$mu["phylo(1 | species)"]     = 0.433551939771
#     sdpars$mu["phylo(0 + x | species)"] = 0.225081189238
#     coef mu = (0.863567925578, -0.292913465640), sigma (log) = -1.53266453997
#     corpars = list()   (no intercept–slope correlation parameter exists)
using DRModels
using Test, Random, LinearAlgebra, Statistics
import Distributions

const _A9C_Y = [
        0.5489613379889846, 0.9906492300376193, 0.6766003844848468, 0.6994577607841, 0.4573770441198175, 0.3326782003235285, -0.019387249463711634, -0.0893911392667144,
        1.5419053682089152, 0.7968189391419382, 1.006204300140601, 0.9617848016117163, 0.5591452629226993, 0.17979792730833827, 0.4267795849918267, 0.34042996971112444,
        2.3544580386763685, 2.5192233262430412, 1.872759765884904, 1.067599118170049, 0.9677124900581207, 1.0204348535502266, 1.0451320977343797, 0.8778022264106069,
        2.3148760662868195, 1.9446684613540375, 1.686445601877821, 1.414485818993537, 1.5554368151325884, 1.6121762778541964, 0.7945523620146276, 1.1152909665163162,
        0.9202453241211029, 1.4101294949587229, 1.3854109977542546, 1.1435378413841657, 1.0082458830703076, 0.8980754079408373, 1.1525667223232716, 0.7772130430662372,
        1.3977650489364981, 1.2413575020506584, 1.4063804591652296, 0.946227902575607, 0.9117797665592442, 1.3405658559524367, 0.38748124960601393, 0.6442334916640862,
        0.8066151857249452, 1.0422160606358857, 0.7072816026757456, 0.6362422536206747, 0.47679027029171733, 0.4793753299258639, 0.5907962776730203, 0.42128369051944015,
        1.3841974140269724, 1.2238258372908535, 1.4866079132268826, 1.0846351065176263, 1.0010264473945392, 1.0856438126216588, 1.0430223901263571, 1.4237733704715823,
        1.1754392833335825, 0.7719626276999584, 1.2304592538043913, 1.028495995708745, 0.8079029027068921, 0.7714400403343702, 0.6178106460970705, 1.085525657206418,
        1.1151497203473553, 1.0345408357794597, 1.0300302559146344, 0.4895007151543085, 0.8974897541999021, 0.7495580760670464, 0.8048401361775338, 0.5919986833701916,
        1.0332313047516415, 1.1078450801717385, 1.2410880138679943, 0.9930257503662039, 0.59542636609263, 0.5019377164941944, -0.2698334199514344, 0.37280997316507936,
        1.337632542386639, 1.1482889622931265, 1.3516633721784919, 1.1914577842447664, 0.967012613586567, 0.6596810602256404, 0.9476035386249831, 0.5816373999077401,
        0.1877112112403018, 0.3716814065984417, 0.0834914707234142, 0.204036418275824, 0.5620356302549238, 0.022015503326112296, -0.02507263669598904, -0.030253030849640414,
        0.1591728699906181, 0.17953987939705546, 0.2775614107394055, -0.09991944280722567, 0.2356635231488146, 0.4345958961548271, 0.16754140244337606, 0.5583649373234745,
        0.8228686830640639, 0.4740283300060935, 0.6898493590984057, 0.8791980441828507, 0.7455490036226479, 0.798879952362636, 0.7879918785826953, 0.6331122130953216,
        1.0019052682543919, 1.0428705447239697, 1.2369672294933598, 1.0540773695452355, 0.8364779848731986, 0.9670025850905937, 1.0708571260268553, 0.8068248466550925,
]

function _a9c_fixture()
    p = 16; m = 8
    phy = random_balanced_tree(p; branch_length = 0.25)
    species = repeat(1:p, inner = m)
    x = repeat(collect(range(-1, 1, length = m)), outer = p)
    return (; y = copy(_A9C_Y), x, species), phy
end

# Independent dense oracle: MvNormal logpdf with V assembled by hand (loops, not
# the fitter's building blocks) at a given θ = [β; log σ; log σₐ; log σ_b].
function _a9c_dense_loglik(θ, y, X, Xσ, species, C)
    n = length(y); p = size(C, 1)
    pμ = size(X, 2); pσ = size(Xσ, 2)
    β = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; σa = exp(θ[pμ+pσ+1]); σb = exp(θ[pμ+pσ+2])
    V = zeros(n, n)
    for i in 1:n, j in 1:n
        cij = C[species[i], species[j]]
        V[i, j] = σa^2 * cij + σb^2 * cij * X[i, 2] * X[j, 2]
    end
    ση = exp.(Xσ * βσ)
    for i in 1:n
        V[i, i] += ση[i]^2
    end
    return Distributions.logpdf(Distributions.MvNormal(X * β, Symmetric(V)), y)
end

@testset "Gaussian two-SD phylogenetic random slope phylo(1 + x | species) (#620)" begin
    data, phy = _a9c_fixture()
    y, x, species = data.y, data.x, data.species
    n = length(y)
    @test phylo_tree_height(phy) ≈ 1.0
    C = DRModels._phylo_correlation(phy)
    @test C[1, 2] ≈ 0.75 && C[1, 3] ≈ 0.5 && C[1, 5] ≈ 0.25 && C[1, 9] ≈ 0.0

    fit = drm(bf(@formula(y ~ x + phylo(1 + x | species))), Gaussian(); data = data, tree = phy)

    @testset "shape: two SDs, no correlation, drmTMB's blocks" begin
        @test fit.converged
        @test all(isfinite, fit.theta)
        @test length(fit.theta) == 5                       # β(2) + log σ + log σₐ + log σ_b
        @test fit.blocks == [:mu => 1:2, :sigma => 3:3, :resd => 4:5]
        @test fit.coefnames == [:mu => ["(Intercept)", "x"], :sigma => ["(Intercept)"],
                                :resd => ["species", "species:x"]]
        @test !any(p -> first(p) === :recov, fit.blocks)   # no intercept–slope correlation parameter
        sds = re_sd(fit)
        @test Set(keys(sds)) == Set([:species, Symbol("species:x")])
        @test all(v -> v > 0, values(sds))
        v = vc(fit)
        @test v[:species] ≈ fill(sds[:species]^2, 1, 1)
        @test v[Symbol("species:x")] ≈ fill(sds[Symbol("species:x")]^2, 1, 1)
        blup = ranef(fit)
        @test Set(keys(blup)) == Set([:species, Symbol("species:x")])
        @test length(blup[:species]) == 16 && length(blup[Symbol("species:x")]) == 16
        @test all(isfinite, sqrt.(diag(fit.vcov)))
    end

    @testset "same target as drmTMB 67703f541 on this fixture (tol 1e-6)" begin
        @test fit.loglik ≈ -19.586599200945 atol = 1e-6
        sds = re_sd(fit)
        @test sds[:species] ≈ 0.433551939771 atol = 1e-6                 # phylo(1 | species)
        @test sds[Symbol("species:x")] ≈ 0.225081189238 atol = 1e-6      # phylo(0 + x | species)
        @test coef(fit, :mu) ≈ [0.863567925578, -0.292913465640] atol = 1e-6
        @test coef(fit, :sigma) ≈ [-1.53266453997] atol = 1e-6
    end

    @testset "term-by-term: fitted logLik equals an independent dense MvNormal at θ̂" begin
        X = [ones(n) x]; Xσ = ones(n, 1)
        @test fit.loglik ≈ _a9c_dense_loglik(fit.theta, y, X, Xσ, species, C) atol = 1e-8
        # A swapped-SD θ is NOT the optimum: the objective must move (guards a
        # fitter that silently uses one SD for both fields).
        θsw = copy(fit.theta); θsw[4], θsw[5] = θsw[5], θsw[4]
        @test _a9c_dense_loglik(θsw, y, X, Xσ, species, C) < fit.loglik - 0.5
        # and the stored objective is that same likelihood
        @test -fit.nll(fit.theta) ≈ fit.loglik atol = 1e-10
        @test -fit.nll(θsw) ≈ _a9c_dense_loglik(θsw, y, X, Xσ, species, C) atol = 1e-8
    end

    @testset "recovery on the known-truth fixture (single draw, 16 tips; stated tolerance)" begin
        # σₐ = 0.55, σ_b = 0.32, σ = 0.22, β = (0.4, −0.25). One draw of a
        # phylogenetically correlated 16-tip field shifts the intercept; the
        # tolerances are the honest single-draw ones, the drmTMB pin above is
        # the sharp check.
        sds = re_sd(fit)
        @test abs(sds[:species] - 0.55) < 0.2
        @test abs(sds[Symbol("species:x")] - 0.32) < 0.15
        @test abs(exp(coef(fit, :sigma)[1]) - 0.22) < 0.05
        @test abs(coef(fit, :mu)[2] - (-0.25)) < 0.15
        @test sds[:species] > sds[Symbol("species:x")]      # ordering of the two fields
    end

    @testset "nested against the intercept-only phylo fit" begin
        fit0 = drm(bf(@formula(y ~ x + phylo(1 | species))), Gaussian(); data = data, tree = phy)
        @test fit0.converged
        @test fit.loglik > fit0.loglik + 1.0     # the slope field is real on this fixture (Δ ≈ 14.6)
    end

    @testset "predict() and heteroscedastic sigma" begin
        pr = predict(fit, data)
        @test pr ≈ fit.means[:mu] atol = 1e-12
        @test pr ≈ [ones(n) x] * coef(fit, :mu) atol = 1e-12
        fit2 = drm(bf(@formula(y ~ x + phylo(1 + x | species)), @formula(sigma ~ x)), Gaussian();
                   data = data, tree = phy)
        @test fit2.converged
        @test fit2.blocks == [:mu => 1:2, :sigma => 3:4, :resd => 5:6]
        X = [ones(n) x]
        @test fit2.loglik ≈ _a9c_dense_loglik(fit2.theta, y, X, X, species, C) atol = 1e-8
        @test fit2.loglik >= fit.loglik - 1e-6           # nested: sigma ~ 1 ⊂ sigma ~ x
    end

    @testset "boundaries stay refused (fail closed)" begin
        data2 = (; y, x, z = randn(Random.Xoshiro(1), n), species)
        # slope-only and multi-slope forms
        @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(0 + x | species))), Gaussian();
                                       data = data2, tree = phy)
        @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(1 + x + z | species))), Gaussian();
                                       data = data2, tree = phy)
        # non-Gaussian: refused here. drmTMB DOES fit this formula on further families,
        # and on Poisson/NegBinomial2 it fits a CORRELATED intercept-slope model
        # (`has_phylo_mu_q2_covariance`) — a different model, so DRModels.jl fails closed.
        yp = Float64.(round.(Int, exp.(0.2 .+ 0.3 .* x)))
        err = try
            drm(bf(@formula(yp ~ x + phylo(1 + x | species))), Poisson();
                data = (; yp, x, species), tree = phy, se = false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("phylo(1 + x | species)", err.msg) && occursin("not implemented", err.msg)
        # REML: the structured mean marker is refused on the generic route (unchanged)
        @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(1 + x | species))), Gaussian();
                                       data = data, tree = phy, method = :REML)
        # a second structured component, an ordinary RE, a sigma RE, sparse algorithms, missing y
        Kid = Matrix{Float64}(I, 16, 16)
        @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(1 + x | species) + relmat(1 | id))),
                                       Gaussian(); data = (; y, x, species, id = species), tree = phy, K = Kid)
        @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(1 + x | species) + (1 | species))),
                                       Gaussian(); data = data, tree = phy)
        @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(1 + x | species)), @formula(sigma ~ 1 + (1 | species))),
                                       Gaussian(); data = data, tree = phy)
        # Routes that RETURN before the slope dispatch and are written for the
        # intercept-only mean field: a structured (phylo) sigma, and the sd()/
        # sd_phylo() location-scale-scale submodels. Each would otherwise fit
        # `phylo(1 | species)` and silently drop the slope (the #620 class).
        @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(1 + x | species)),
                                          @formula(sigma ~ phylo(1 | species))),
                                       Gaussian(); data = data, tree = phy)
        @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(1 + x | species)),
                                          @formula(sigma ~ 1), @formula(sd_phylo(species) ~ x)),
                                       Gaussian(); data = data, tree = phy)
        @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(1 + x | species)),
                                          @formula(sigma ~ 1), @formula(sd(species) ~ x)),
                                       Gaussian(); data = data, tree = phy)
        @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(1 + x | species))), Gaussian();
                                       data = data, tree = phy, algorithm = :sparse_lbfgs)
        ymiss = copy(y); ymiss[3] = NaN
        @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(1 + x | species))), Gaussian();
                                       data = (; y = ymiss, x, species), tree = phy)
        # the parametric-bootstrap marginal simulator is not built for this fit
        @test_throws ArgumentError DRModels._marginal_simulator(fit, data; tree = phy)
    end
end
