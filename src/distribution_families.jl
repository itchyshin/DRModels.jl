# distribution_families.jl — DRM family markers backed by Distributions.jl types
# where both packages share a name (#1534). Re-exporting the Distributions types
# lets `using DRModels, Distributions` resolve one binding for simulation and for
# `drm(..., Poisson())` / `Gamma()` / `Beta()` / `LogNormal()`.

import Distributions: Poisson, Gamma, Beta, LogNormal

export Poisson, Gamma, Beta, LogNormal