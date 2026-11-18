# Issue #979 -- `sd_phylo` must be on ONE scale whatever the sigma formula.
#
# On ade4::carni70 (ultrametric, height 53.8) the Gaussian
# `lsize ~ 1 + lrange + phylo(1 | species)` fit with `sigma ~ 1` (the sparse
# phylo-mean route) reported log sd_phylo = -1.2917751 on the RAW branch-length
# scale (and warned), while `sigma ~ 1 + lrange` (the dense route) reported the
# tip-CORRELATION scale. drmTMB (`vcv(tree, corr = TRUE)`) is on the correlation
# scale for both: 0.7008616 for `sigma ~ 1`, 0.5176010 for `sigma ~ 1 + lrange`.
# Owner decision (11 Oct 2026, option a): the correlation scale on every route.
# The sparse route now fits on the unit-height tree (for an ultrametric tree
# that IS the correlation matrix) and records `phylo_scale = :correlation`.
#
# Reference values: drmTMB 0.7.1 as quoted in issue #979 (carni70 data and tree
# vendored in test/fixtures/ade4, see its README).
using DRModels
using Test, LinearAlgebra, Logging

const _D979 = DRModels
const _FX979 = joinpath(@__DIR__, "fixtures", "ade4")
const _H979 = 53.8

function _data979()
    lines = readlines(joinpath(_FX979, "carni70.csv"))
    rows = [split(l, ",") for l in lines[2:end]]
    return (species = [String(strip(r[1], '"')) for r in rows],
            lsize = [parse(Float64, r[2]) for r in rows],
            lrange = [parse(Float64, r[3]) for r in rows])
end

const _NWK979 = String(strip(read(joinpath(_FX979, "carni70.nwk"), String)))
# The same tree with every branch length divided by its height.
const _NWK979_UNIT = replace(_NWK979, r":[0-9.eE+-]+" =>
                             m -> ":" * repr(parse(Float64, m[2:end]) / _H979))

_fit979(sig, nwk; kw...) = drm(bf(@formula(lsize ~ 1 + lrange + phylo(1 | species)), sig),
                               Gaussian(); data = _data979(), tree = nwk, kw...)

@testset "#979: sparse sigma ~ 1 route reports sd_phylo on the correlation scale" begin
    # Clear the once-per-tree warning registry so the "no warning" check is not
    # satisfied merely because an earlier test in this session already warned.
    empty!(_D979._PHYLO_HEIGHT_WARNED)
    local f1
    @test_logs min_level = Logging.Warn begin
        f1 = _fit979(@formula(sigma ~ 1), _NWK979)
    end
    @test f1.phylo_scale === :correlation
    # drmTMB: log_sd_phylo 0.7008616, logLik -104.1176.
    @test f1.theta[end] ≈ 0.7008616 atol = 1e-6
    @test f1.loglik ≈ -104.11757559 atol = 1e-6
    @test f1.converged
    # `re_sd` reads the same number, and `scale = :drmtmb` is a no-op (no tree needed).
    @test re_sd(f1)[:species] ≈ exp(0.7008616) rtol = 1e-6
    @test re_sd(f1; scale = :drmtmb) == re_sd(f1)

    # The same model on the unit-height tree: nothing may change.
    fu = _fit979(@formula(sigma ~ 1), _NWK979_UNIT)
    @test fu.phylo_scale === :correlation
    @test fu.theta ≈ f1.theta atol = 1e-6
    @test fu.loglik ≈ f1.loglik atol = 1e-8

    # `sigma ~ 1 + lrange` (dense route) is unchanged and on the same scale:
    # drmTMB 0.5176010, logLik -102.5177.
    f2 = _fit979(@formula(sigma ~ 1 + lrange), _NWK979)
    @test f2.phylo_scale === :correlation
    @test f2.theta[end] ≈ 0.5176010 atol = 1e-6
    @test f2.loglik ≈ -102.51774363 atol = 1e-6

    # Cross-route identity on a model both routes can fit (`sigma ~ 1`): the
    # dense GLS route has always reported the correlation scale.
    fd = _fit979(@formula(sigma ~ 1), _NWK979; algorithm = :gls)
    @test fd.phylo_scale === :correlation
    @test f1.theta[end] ≈ fd.theta[end] atol = 1e-4
    @test f1.loglik ≈ fd.loglik atol = 1e-6
end

@testset "#979: the EM and REML variants of the sparse route follow the same convention" begin
    fr = _fit979(@formula(sigma ~ 1), _NWK979; method = :REML)
    fru = _fit979(@formula(sigma ~ 1), _NWK979_UNIT; method = :REML)
    @test fr.phylo_scale === :correlation
    @test fr.theta ≈ fru.theta atol = 1e-6
    @test fr.reml_loglik ≈ fru.reml_loglik atol = 1e-8
end

@testset "#979: a non-ultrametric tree keeps the raw scale and its warning" begin
    # A tip-wise correlation is not one scalar rescaling, so this route cannot map
    # a non-ultrametric tree onto the correlation scale; it stays raw and says so.
    nwk = "((A:1.0,B:2.0):1.0,(C:0.5,D:3.0):2.0);"
    phy = augmented_phy(nwk)
    @test !_D979._phylo_tree_is_ultrametric(phy)
    @test _D979._phylo_tree_is_ultrametric(augmented_phy(_NWK979))
    phy2, scale = _D979._phylo_mean_correlation_tree(phy)
    @test scale === :covariance
    @test phy2 === phy
end
