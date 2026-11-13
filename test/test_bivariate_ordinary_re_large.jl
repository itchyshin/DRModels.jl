# Bivariate Gaussian with an ordinary `(1 | p | g)`: larger data, pinned to drmTMB 0.8.1.
#
# Regression: the first line-search trial of the optimiser can step the log scale
# parameters to about -370; exp(-ls) then overflows, the 2x2 group matrix
# I + L'AL becomes non-finite, and `inv(Symmetric(...))` threw "matrix contains
# Infs or NaNs", aborting the fit at n of about 1200 (ML and REML) although
# drmTMB fits these. The objective now returns a non-finite value for such a trial
# so the line search backs off. Data and reference: test/fixtures/biv_ordinary_re/
# (gen_review_data.R, make_reference_review.R), simulated truth group SD 0.8 / 0.6,
# group corr 0.5, residual rho12 0.3, all interior. Same fixed bounds as the
# original file.
include(joinpath(@__DIR__, "biv_ordinary_re_helpers.jl"))

@testset "bivariate ordinary (1 | p | g), larger n, pinned to drmTMB" begin
    with_logger(NullLogger()) do
        d1200 = _bre_data("large_1200.csv")      # 120 groups of 10
        @testset "n = 1200, ML" begin
            _bre_check("large1200_ml", drm(_BRE_F0, Gaussian(); data = d1200), 7)
        end
        @testset "n = 1200, REML" begin
            _bre_check("large1200_reml", drm(_BRE_F0, Gaussian(); data = d1200, method = :REML), 7)
        end
        @testset "n = 2400 (150 groups of 16), ML" begin
            _bre_check("large2400_ml", drm(_BRE_F0, Gaussian(); data = _bre_data("large_2400.csv")), 7)
        end
    end
end
