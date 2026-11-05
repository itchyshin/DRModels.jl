# re_sd(), vc(), heritability() and the boundary diagnostics refuse fits that
# model the random-effect SD with covariates. The guard must recognise every
# SD-regression block name, not only `:sd` and `:sd_phylo`: the two-group
# Binomial route uses `:sd_g` / `:sd_h` and the structured and correlated-block
# routes use typed `:sdtarget_<id>` blocks. Missing them let re_sd() decode a
# one-element correlation `:recov` block as a Cholesky triple (BoundsError in
# convergence_status() on Binomial animal q2 fits).
using DRModels, Test

@testset "SD-regression block guard" begin
    guard = DRModels._has_sd_regression_block
    blk(names...) = (; blocks = [n => (1:1) for n in names])
    @test guard(blk(:mu, :sd))
    @test guard(blk(:mu, :sd_phylo))
    @test guard(blk(:mu, :sd_g, :sd_h))
    @test guard(blk(:mu, :sdtarget_6162, :sdtarget_6364, :recov))
    @test !guard(blk(:mu, :resd))
    @test !guard(blk(:mu, :sigma, :resd, :recov))
    @test !guard(blk(:mu))
end
