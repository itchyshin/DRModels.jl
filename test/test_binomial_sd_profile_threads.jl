using DRModels, Test

@testset "Binomial SD profile threading identifies structured targets" begin
    emptydict() = Dict{Symbol,Vector{Float64}}()
    dummyfit(family, blocks, formula) = DRModels._withformula(
        DRModels.DrmFit(
            family,
            Pair{Symbol,UnitRange{Int}}[blocks...],
            Pair{Symbol,Vector{String}}[], Float64[], zeros(0, 0), 0.0, 0,
            false, emptydict(), emptydict(), emptydict(),
        ),
        formula,
    )

    iid = dummyfit(
        Binomial(), [:sd => (1:2)],
        bf(@formula(y ~ (1 | g)), @formula(sd(g) ~ z)),
    )
    fit_phylo = dummyfit(
        Binomial(), [:sd_phylo => (1:2)],
        bf(@formula(y ~ phylo(1 | species)),
           @formula(sd(species, phylogenetic) ~ z)),
    )
    fit_animal = dummyfit(
        Binomial(), [:sd_animal => (1:2)],
        bf(@formula(y ~ animal(1 | g)), @formula(sd(g, animal) ~ z)),
    )
    fixed = dummyfit(Binomial(), [:mu => (1:2)], bf(@formula(y ~ x)))
    gaussian = dummyfit(Gaussian(), [:sd_phylo => (1:2)], nothing)

    @test !DRModels._binomial_sd_threaded_profile_unsupported(iid)
    @test DRModels._binomial_sd_threaded_profile_unsupported(fit_phylo)
    @test DRModels._binomial_sd_threaded_profile_unsupported(fit_animal)
    @test !DRModels._binomial_sd_threaded_profile_unsupported(fixed)
    @test !DRModels._binomial_sd_threaded_profile_unsupported(gaussian)
end
