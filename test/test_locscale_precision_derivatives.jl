using DRModels
using Test, TOML, SparseArrays, LinearAlgebra

# Generated reference values come from an independently implemented, whitened
# Gamma Laplace objective at the original four failed profile nuisance states.
# The reference uses 128/256-bit inner solves and Richardson directional
# derivatives, not production's inverse-covariance derivative formula.
@testset "Gamma covariance gradient at retained profile failures" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "locscale_precision",
                                      "locscale_gamma_l21.toml"))
    @test fixture["G"] == 4
    @test length(fixture["y"]) == length(fixture["x"]) == length(fixture["gidx"]) == 32
    @test length(fixture["cases"]) == 4
    @test fixture["provenance"]["expansion_output_sha256"] ==
        "d650f848231c2d7da2d135092720c96c5875dc83621d3e4a45404e6e6bd0baf2"
    Xmu = hcat(ones(32), fixture["x"])
    Xpsi = ones(32, 1)
    Q = sparse(1.0I, 4, 4)
    for case in fixture["cases"]
        @testset "$(case["label"])" begin
            gradient = DRModels._ls_marginal_grad(Val(:gamma), fixture["y"], Xmu, Xpsi,
                fixture["gidx"], fixture["G"], Q, case["theta"])
            @test all(isfinite, gradient)
            # Absolute accuracy must respect the existing profile stationarity
            # tolerance even when the covariance derivative is near zero.
            @test isapprox(gradient[5], case["expected_l21"]; rtol=0.0, atol=1e-7)
            # Independently difference the inverse covariance in high precision
            # to exercise all three derivatives, including the diagonal ones.
            lambda = case["theta"][4:6]
            derivatives = DRModels._ls_precision_derivatives(lambda)
            setprecision(BigFloat, 256) do
                h = big"1e-8"
                for k in 1:3
                    function shifted_inverse(offset)
                        point = BigFloat.(lambda)
                        point[k] += offset
                        L = BigFloat[exp(point[1]) 0; point[2] exp(point[3])]
                        return inv(L * transpose(L))
                    end
                    reference = (-shifted_inverse(2 * h) + 8 * shifted_inverse(h) -
                                 8 * shifted_inverse(-h) + shifted_inverse(-2 * h)) / (12 * h)
                    relative_error = maximum(abs, BigFloat.(derivatives[k]) - reference) /
                                     maximum(abs, reference)
                    @test relative_error < big"1e-13"
                end
            end
        end
    end
end

# Squaring the shared exponential before multiplying by zero can overflow even
# when every derivative entry is representable. The reference is evaluated in
# BigFloat from the exact diagonal-covariance derivatives at c = 0.
@testset "finite precision derivatives at zero covariance" begin
    lambda = [-180.0, 0.0, -180.0]
    derivatives = DRModels._ls_precision_derivatives(lambda)
    @test all(d -> all(isfinite, d), derivatives)
    setprecision(BigFloat, 256) do
        a = exp(big"360")
        cross = exp(big"540")
        reference = ([-2*a big"0"; big"0" big"0"],
                     [big"0" -cross; -cross big"0"],
                     [big"0" big"0"; big"0" -2*a])
        for k in 1:3
            @test isapprox(BigFloat.(derivatives[k]), reference[k]; rtol=big"1e-14", atol=0)
        end
    end
end
