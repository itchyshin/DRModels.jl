# Drawing-layer entry point. The METHOD lives in the `DRModelsMakieExt` package
# extension (`ext/DRModelsMakieExt.jl`), which loads only when `Makie` (e.g.
# `using CairoMakie`) and `AlgebraOfGraphics` are in scope — so `/src` stays
# dependency-free. Prepare-data remains in `visualization.jl`; this file is the
# stub + thin `plot_*` aliases. No engine computation happens here.

"""
    drm_figure(data; kind = <inferred>, kwargs...)

Draw a DRModels plotting-layer figure from a preparer `NamedTuple`
([`profile_curve`](@ref) / [`parameter_surface`](@ref) / [`corpairs_data`](@ref)).
**STUB:** the drawing method is provided by the `DRModelsMakieExt` package extension,
which activates only when a Makie backend and AlgebraOfGraphics are loaded
(`using CairoMakie, AlgebraOfGraphics` / `using GLMakie, AlgebraOfGraphics`).
Without them, calling this throws a `MethodError` asking you to load the
plotting weak dependencies.

Supported `kind`s:

- `:profile` ← [`profile_curve`](@ref) — profile deviance with Florence's
  **Confidence Eye** (pale compatibility region where deviance ≤ cutoff, darker
  outline, hollow point estimate)
- `:parameter_surface` ← [`parameter_surface`](@ref) — 2-D profile-deviance
  heatmap / surface
- `:corpairs` ← [`corpairs_data`](@ref) — residual `ρ12` summary (constant point
  or per-observation scatter)

Drawing only — no estimation, no engine computation. Default CI does **not**
load Makie; stub tests assert `isempty(methods(drm_figure))` without the weakdeps.
"""
function drm_figure end

"""
    plot_profile(fit, k; npoints = 41, span = 3.0, level = 0.95, kwargs...)

Thin drmTMB-named alias: [`profile_curve`](@ref) then [`drm_figure`](@ref)
(`kind = :profile`). Requires Makie + AlgebraOfGraphics (via `DRModelsMakieExt`).
"""
function plot_profile(fit, k; npoints::Int=41, span::Real=3.0, level::Real=0.95, kwargs...)
    return drm_figure(profile_curve(fit, k; npoints, span, level); kind=:profile, kwargs...)
end

"""
    plot_parameter_surface(fit, k1, k2; npoints = 25, span = 3.0, kwargs...)

Thin drmTMB-named alias: [`parameter_surface`](@ref) then [`drm_figure`](@ref)
(`kind = :parameter_surface`). Requires Makie + AlgebraOfGraphics (via `DRModelsMakieExt`).
"""
function plot_parameter_surface(fit, k1, k2; npoints::Int=25, span::Real=3.0, kwargs...)
    return drm_figure(parameter_surface(fit, k1, k2; npoints, span); kind=:parameter_surface, kwargs...)
end

"""
    plot_corpairs(fit; kwargs...)

Thin drmTMB-named alias: [`corpairs_data`](@ref) then [`drm_figure`](@ref)
(`kind = :corpairs`). Requires Makie + AlgebraOfGraphics (via `DRModelsMakieExt`).
"""
function plot_corpairs(fit; kwargs...)
    return drm_figure(corpairs_data(fit); kind=:corpairs, kwargs...)
end
