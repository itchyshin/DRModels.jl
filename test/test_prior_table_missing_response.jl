using Test, DRModels

# A fit with priors and a `missing` response must stop with a clear
# ArgumentError, the same refusal a `NaN` response already gets. It used to end
# in a raw MethodError (Missing to Float64) inside the response check.
module PriorMissingResponseCase
using Test, DRModels

const n = 40
const grp = repeat(string.("s", 1:8), inner = 5)
const obs = repeat(string.("o", 1:5), 8)
const x = [sin(0.7i) + 0.1i / n for i in 1:n]
# Group and observer effects are clearly non-zero, so the two-intercept fits stay off the boundary.
const y_full = [1.0 + 0.5x[i] + 0.9cos(2.4parse(Int, grp[i][2:end])) + 0.6sin(3.1parse(Int, obs[i][2:end])) +
                0.25cos(1.3i) + 0.2sin(2.1i) for i in 1:n]
const gone = [3, 17, 30]
const y_miss = Union{Missing,Float64}[i in gone ? missing : y_full[i] for i in 1:n]
const y_nan = [i in gone ? NaN : y_full[i] for i in 1:n]
const bin_full = [sin(1.9i) + 0.8x[i] > 0.2 ? 1 : 0 for i in 1:n]
const bin_miss = Union{Missing,Int}[i in (2, 9) ? missing : bin_full[i] for i in 1:n]

dat(y) = (y = y, x = x, g = grp, h = obs)

# Phylogenetic SD regression with fixed-effect priors (a separate fitting route).
const phy = DRModels.augmented_phy("((a:0.5,b:0.5):0.5,(c:0.5,d:0.5):0.5);")
const phy_full = (y = [.2, .7, .35, 1.1, .8, 1.25, 1.4, 1.1, 1.9, 1.5, 2.15, .5],
                  x = [-1.2, -.8, -.7, -.2, .1, .15, .4, .6, .8, 1.0, 1.2, -.3],
                  sp = ["a", "a", "b", "b", "b", "c", "c", "d", "d", "d", "d", "a"],
                  z = [-1.0, -1.0, -.35, -.35, -.35, .4, .4, 1.0, 1.0, 1.0, 1.0, -1.0])
phy_data(y) = merge(phy_full, (y = y,))
const phy_miss = phy_data(Union{Missing,Float64}[i in (2, 7) ? missing : phy_full.y[i] for i in 1:12])
const phy_nan = phy_data([i in (2, 7) ? NaN : phy_full.y[i] for i in 1:12])
const phy_form = bf(@formula(y ~ x + phylo(1 | sp)), @formula(sigma ~ x), @formula(sd(sp, phylogenetic) ~ z))
const phy_table = (prior = ["normal(.1,.7)", "normal(-.2,.5)"], class = ["fixef", "fixef"],
                   coef = ["x", "x"], dpar = ["mu", "sigma"])
const phy_legacy = drm_priors(drm_prior_normal(dpar = "mu", coef = "x", mean = .1, sd = .7),
                              drm_prior_normal(dpar = "sigma", coef = "x", mean = -.2, sd = .5))
const keep = setdiff(1:n, gone)
const complete = (y = y_full[keep], x = x[keep], g = grp[keep], h = obs[keep])

const fixed_form = bf(@formula(y ~ x))
const two_form = bf(@formula(y ~ x + (1 | g) + (1 | h)))
const fixed_table = (prior = ["normal(0,2)"], class = ["fixef"], coef = ["x"])
const sd_table = (prior = ["gamma(1,2)"], class = ["ranef_sd"], coef = [""])
const flat_table = (prior = ["ridge(0)"], class = ["fixef"], coef = ["x"])
const fixed_legacy = drm_priors(drm_prior_normal(dpar = "mu", coef = "x", mean = 0, sd = 2))
const sd_legacy = drm_priors(drm_prior_sd_gamma(dpar = "mu", group = "g", shape = 2, rate = 2),
                             drm_prior_sd_gamma(dpar = "mu", group = "h", shape = 2, rate = 2))

# The refusal must be an ArgumentError that names the response, never a MethodError.
function refused(thunk)
    try
        thunk()
        return (false, "no error")
    catch e
        return (e isa ArgumentError && occursin("complete responses", e.msg), sprint(showerror, e))
    end
end

@testset "prior fit with a missing response is refused, not a MethodError" begin
    @testset "Gaussian fixed effects" begin
        for pri in (fixed_table, fixed_legacy)
            ok, msg = refused(() -> drm(fixed_form, Gaussian(); data = dat(y_miss), priors = pri))
            @test ok
            ok_nan, msg_nan = refused(() -> drm(fixed_form, Gaussian(); data = dat(y_nan), priors = pri))
            @test ok_nan
        end
    end
    @testset "Gaussian two random intercepts" begin
        for pri in (fixed_table, fixed_legacy, sd_table, sd_legacy)
            ok, msg = refused(() -> drm(two_form, Gaussian(); data = dat(y_miss), priors = pri))
            @test ok
        end
    end
    @testset "preview" begin
        @test refused(() -> resolve_priors(fixed_form, Gaussian(); data = dat(y_miss), priors = fixed_table))[1]
        @test refused(() -> resolve_priors(two_form, Gaussian(); data = dat(y_miss), priors = sd_table))[1]
    end
    @testset "binomial" begin
        for pri in (fixed_table, fixed_legacy)
            ok, msg = refused(() -> drm(bf(@formula(k ~ x)), Binomial(); data = (k = bin_miss, x = x), priors = pri))
            @test ok
        end
    end
    @testset "phylogenetic SD regression with fixed-effect priors" begin
        for pri in (phy_table, phy_legacy)
            @test refused(() -> drm(phy_form, Gaussian(); data = phy_miss, tree = phy, priors = pri))[1]
        end
        @test refused(() -> drm(phy_form, Gaussian(); data = phy_nan, tree = phy, priors = phy_table))[1]
        @test refused(() -> resolve_priors(phy_form, Gaussian(); data = phy_miss, priors = phy_table))[1]
        fit = drm(phy_form, Gaussian(); data = phy_full, tree = phy, priors = phy_table)
        @test refused(() -> update(fit, phy_form; data = phy_miss, tree = phy))[1]
    end
    @testset "response kept with miss_control(response = \"include\")" begin
        inc = miss_control(response = "include")
        @test refused(() -> drm(fixed_form, Gaussian(); data = dat(y_miss), priors = fixed_table, missing = inc))[1]
        @test refused(() -> drm(phy_form, Gaussian(); data = phy_miss, tree = phy, priors = phy_table, missing = inc))[1]
    end
    @testset "update with a missing response column" begin
        fit = drm(fixed_form, Gaussian(); data = dat(y_full), priors = fixed_table)
        @test refused(() -> update(fit, fixed_form; data = dat(y_miss)))[1]
    end
end

@testset "no-prior fit drops the rows; priors on the complete rows agree with it" begin
    # Without priors, missing responses are dropped and the fit equals the fit
    # on the complete rows. A flat prior table on the complete rows is the same
    # model, so it reproduces that fit (this is the advice the refusal gives).
    for form in (fixed_form, two_form)
        with_missing = drm(form, Gaussian(); data = dat(y_miss))
        on_complete = drm(form, Gaussian(); data = complete)
        @test nobs(with_missing) == nobs(on_complete) == length(keep)
        @test coef(with_missing) ≈ coef(on_complete) rtol = 1e-8 atol = 1e-8
        @test loglikelihood(with_missing) ≈ loglikelihood(on_complete) rtol = 1e-9
        flat = drm(form, Gaussian(); data = complete, priors = flat_table)
        @test nobs(flat) == length(keep)
        @test coef(flat) ≈ coef(on_complete) rtol = 1e-5 atol = 1e-5
        @test loglikelihood(flat) ≈ loglikelihood(on_complete) rtol = 1e-6
    end
    # Complete data with a prior still fits as before.
    full = drm(fixed_form, Gaussian(); data = dat(y_full), priors = fixed_table)
    @test nobs(full) == n
    @test all(isfinite, coef(full))
end
end
