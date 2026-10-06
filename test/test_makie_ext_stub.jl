# DRModelsMakieExt drawing stub (#336). Mirror of HSquared's hsquared_figure CI gate:
# `drm_figure` is a STUB in /src; the drawing METHOD lives in ext/DRModelsMakieExt.jl
# and loads only when Makie + AlgebraOfGraphics are in scope. Default CI must NOT
# depend on Makie — these tests assert the stub stays method-less.
#
# TRAP — READ THIS BEFORE TRUSTING A GREEN CI RUN. `isempty(methods(...))` passes
# when the extension is absent (the intended default). It does NOT prove the
# drawing layer renders correctly. Optional local CairoMakie smoke is opt-in only
# and is never claimed as CI evidence.
using DRModels
using Test

@testset "drm_figure drawing stub (DRModelsMakieExt weak-dep, #336)" begin
    @test drm_figure isa Function
    @test isempty(methods(drm_figure))          # stub: no methods without Makie+AoG
    @test_throws MethodError drm_figure((
        x=[0.0, 1.0],
        deviance=[1.0, 0.0],
        estimate=1.0,
        cutoff=3.84,
        k=1,
        param=:mu,
        coef="x",
        level=0.95,
    ))
    @test_throws MethodError drm_figure((
        x=[0.0, 1.0],
        y=[0.0, 1.0],
        z=[0.0 1.0; 1.0 2.0],
        k1=1,
        k2=2,
    ))
    @test_throws MethodError drm_figure((rho=[0.3], constant=true))

    # Thin plot_* aliases prepare then call drm_figure — also MethodError without ext.
    @test plot_profile isa Function
    @test plot_parameter_surface isa Function
    @test plot_corpairs isa Function
end
