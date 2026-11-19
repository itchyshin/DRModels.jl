using DRModels, Test, Random

# #1387: Poisson, NB2 and Gaussian fits whose random-effect shape has no fresh-draw simulator used to
# bootstrap with the random effects at zero and collapse the variance-component interval without a
# message. They now refuse by name (SHINICHI-0900-1119), like Gamma, Beta, Student and LogNormal.
function _d1387(seed)
    r = Random.Xoshiro(seed); n = 600; G = 30
    gi = rand(r, 1:G, n); x = randn(r, n); z = randn(r, n)
    b0 = 0.4 .* randn(r, G); b1 = 0.3 .* randn(r, G)
    eta = 0.3 .+ 0.3 .* x .+ b0[gi] .+ b1[gi] .* z
    lam = exp.(eta)
    ypois = [_pois(r, l) for l in lam]
    ynb = [_pois(r, l * 2 * rand(r, 1:3) / 2) for l in lam]
    ygau = eta .+ 0.5 .* randn(r, n)
    return (; x, z, g = string.("g", gi), ypois, ynb, ygau)
end
function _pois(r, lam)
    u = rand(r); k = 0; p = exp(-lam); c = p
    while u > c && k < 1000; k += 1; p *= lam / k; c += p; end
    return k
end

@testset "#1387 (1 + z | g) refuses by name for Poisson, NB2, Gaussian" begin
    d = _d1387(3)
    for (fname, fam, resp) in (("Poisson", Poisson(), :ypois), ("NegBinomial2", NegBinomial2(), :ynb),
                               ("Gaussian", Gaussian(), :ygau))
        fo = Core.eval(@__MODULE__, Meta.parse("bf(@formula($resp ~ x + (1 + z | g)))"))
        fit = drm(fo, fam; data = d)
        for call in (() -> bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)),
                     () -> bootstrap_result(fit; data = d, B = 2, rng = Random.Xoshiro(1)),
                     () -> simulate(fit; data = d, rng = Random.Xoshiro(1)))
            err = try; call(); nothing; catch e; e; end
            @test err isa ArgumentError
            err isa ArgumentError || continue
            msg = sprint(showerror, err)
            @test occursin("no fresh random-effect draw", msg)
            @test occursin("for $fname() with a correlated random intercept and slope ((1 + z | g))", msg)
        end
        @test length(simulate(fit; data = d, re = :zero, rng = Random.Xoshiro(1))) == length(d.x)
    end
end

@testset "#1387 shapes that keep a fresh draw are not refused" begin
    d = _d1387(3)
    fit = drm(bf(@formula(ygau ~ x + (0 + z | g))), Gaussian(); data = d)
    @test length(simulate(fit; data = d, rng = Random.Xoshiro(1))) == length(d.x)
    fit = drm(bf(@formula(ypois ~ x + (1 | g))), Poisson(); data = d)
    @test length(simulate(fit; data = d, rng = Random.Xoshiro(1))) == length(d.x)
end
