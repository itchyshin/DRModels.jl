# Visualization

!!! note "Status — Reference"
    Mirrors drmTMB's [Visualization](https://itchyshin.github.io/drmTMB/reference/index.html) (3 in drmTMB). DRModels.jl keeps the base package plotting-dependency-free: these return the *data* a plot needs — a profile-deviance grid, a correlation summary — so any backend can render them. Drawing is optional via the **`DRModelsMakieExt`** package extension (Makie + AlgebraOfGraphics weakdeps); load a Makie backend and AlgebraOfGraphics to activate [`drm_figure`](@ref). Default CI does **not** draw figures — it only gates the method-less stub.

## Plotting data providers

```@docs
profile_curve
parameter_surface
corpairs_data
```

## Drawing layer (optional Makie extension)

```@docs
drm_figure
plot_profile
plot_parameter_surface
plot_corpairs
```

To draw locally (not in default CI):

```julia
using CairoMakie, AlgebraOfGraphics   # activates DRModelsMakieExt
pc = profile_curve(fit, 2)
fig = drm_figure(pc; kind = :profile)   # Confidence Eye on the profile interval
# or: plot_profile(fit, 2)
```
