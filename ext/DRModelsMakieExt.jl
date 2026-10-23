module DRModelsMakieExt

# Makie/AoG drawing extension for DRModels's backend-free visualization preparers.
# Loads only when a Makie backend and AlgebraOfGraphics are in scope
# (`using CairoMakie, AlgebraOfGraphics`). Arc 0 drawings are mostly raw Makie;
# AlgebraOfGraphics is in the extension gate to match the HSquared house pattern.
# Florence's Confidence Eye contract applies to `:profile`.

using DRModels
using AlgebraOfGraphics
using Makie
using Printf: @sprintf
import DRModels: drm_figure

function _infer_kind(d::NamedTuple)
    hasproperty(d, :deviance) && hasproperty(d, :cutoff) && hasproperty(d, :estimate) &&
        return :profile
    hasproperty(d, :z) && hasproperty(d, :k1) && hasproperty(d, :k2) &&
        return :parameter_surface
    hasproperty(d, :rho) && hasproperty(d, :constant) && return :corpairs
    throw(ArgumentError(
        "drm_figure: cannot infer the figure kind from this data; pass " *
        "`kind = :profile | :parameter_surface | :corpairs`",
    ))
end

function drm_figure(data::NamedTuple; kind::Symbol=_infer_kind(data), kwargs...)
    kind === :profile && return _profile(data; kwargs...)
    kind === :parameter_surface && return _parameter_surface(data; kwargs...)
    kind === :corpairs && return _corpairs(data; kwargs...)
    throw(ArgumentError(
        "drm_figure: unknown kind :$kind (supported: :profile, :parameter_surface, :corpairs)",
    ))
end

# ── :profile — profile deviance + Confidence Eye ────────────────────────────────
# Pale compatibility region (deviance ≤ cutoff), darker outline, hollow estimate.
function _profile(
    d::NamedTuple;
    title="Profile likelihood",
    hue=:steelblue,
    kwargs...,
)
    x = Float64.(d.x)
    dev = Float64.(d.deviance)
    est = Float64(d.estimate)
    cutoff = Float64(d.cutoff)
    lo, hi = _profile_interval(x, dev, cutoff, est)
    param = hasproperty(d, :param) ? string(d.param) : "θ"
    coef = hasproperty(d, :coef) ? string(d.coef) : ""
    level = hasproperty(d, :level) ? d.level : 0.95
    xlab = isempty(coef) ? param : "$param · $coef"
    caveat = @sprintf(
        "Confidence Eye: pale region = %.0f%% profile interval [%.3g, %.3g]; hollow = MLE",
        100 * level,
        lo,
        hi,
    )

    fig = Figure(; size=(720, 420))
    ax = Axis(
        fig[1, 1];
        title=title,
        subtitle=caveat,
        xlabel=xlab,
        ylabel="profile deviance 2(ℓ̂ − ℓ)",
    )
    # Pale fill under the curve inside the compatibility region.
    inside = (x .>= lo) .& (x .<= hi)
    if any(inside)
        xs = x[inside]
        ys = min.(dev[inside], cutoff)
        band!(ax, xs, zero(xs), ys; color=(hue, 0.18))
        lines!(ax, xs, ys; color=hue, linewidth=1.5)          # darker upper outline
        lines!(ax, [lo, hi], [0.0, 0.0]; color=hue, linewidth=1.5)
        lines!(ax, [lo, lo], [0.0, min(cutoff, _interp_dev(x, dev, lo))]; color=hue, linewidth=1.2)
        lines!(ax, [hi, hi], [0.0, min(cutoff, _interp_dev(x, dev, hi))]; color=hue, linewidth=1.2)
    end
    lines!(ax, x, dev; color=:black, linewidth=2.0)
    hlines!(ax, [cutoff]; color=(:gray, 0.7), linestyle=:dash)
    # Hollow point estimate at the MLE (deviance ≈ 0).
    scatter!(
        ax,
        [est],
        [0.0];
        marker=:circle,
        color=:white,
        strokecolor=hue,
        strokewidth=1.8,
        markersize=12,
    )
    return fig
end

function _profile_interval(x::AbstractVector, dev::AbstractVector, cutoff::Real, est::Real)
    n = length(x)
    n == 0 && return (est, est)
    if all(d -> d <= cutoff + 1e-12, dev)
        return (Float64(x[1]), Float64(x[end]))
    end
    # Leftmost / rightmost grid points with deviance ≤ cutoff, then linear refine.
    inside = findall(d -> d <= cutoff + 1e-12, dev)
    isempty(inside) && return (est, est)
    iL, iR = first(inside), last(inside)
    lo = Float64(x[iL])
    hi = Float64(x[iR])
    if iL > 1
        lo = _cross(x[iL - 1], dev[iL - 1], x[iL], dev[iL], cutoff)
    end
    if iR < n
        hi = _cross(x[iR], dev[iR], x[iR + 1], dev[iR + 1], cutoff)
    end
    return (lo, hi)
end

function _cross(x0, y0, x1, y1, c)
    dy = y1 - y0
    abs(dy) < 1e-15 && return x0
    t = (c - y0) / dy
    return x0 + t * (x1 - x0)
end

function _interp_dev(x, dev, t)
    i = searchsortedlast(x, t)
    i < 1 && return Float64(dev[1])
    i >= length(x) && return Float64(dev[end])
    x0, x1 = x[i], x[i + 1]
    abs(x1 - x0) < 1e-15 && return Float64(dev[i])
    w = (t - x0) / (x1 - x0)
    return (1 - w) * Float64(dev[i]) + w * Float64(dev[i + 1])
end

# ── :parameter_surface — 2-D profile deviance ───────────────────────────────────
function _parameter_surface(
    d::NamedTuple;
    title="Profile deviance surface",
    kwargs...,
)
    x = Float64.(d.x)
    y = Float64.(d.y)
    z = Matrix{Float64}(d.z)
    fig = Figure(; size=(720, 480))
    ax = Axis(
        fig[1, 1];
        title=title,
        subtitle="profile deviance 2(ℓ̂ − ℓ); χ²₂ contours = joint regions",
        xlabel="θ[$(d.k1)]",
        ylabel="θ[$(d.k2)]",
    )
    hm = heatmap!(ax, x, y, z; colormap=:viridis)
    Colorbar(fig[1, 2], hm; label="profile deviance")
    # Hollow marker near the MLE (grid minimum).
    ij = argmin(z)
    scatter!(
        ax,
        [x[ij[1]]],
        [y[ij[2]]];
        marker=:circle,
        color=:white,
        strokecolor=:black,
        strokewidth=1.6,
        markersize=11,
    )
    return fig
end

# ── :corpairs — residual ρ12 summary ────────────────────────────────────────────
function _corpairs(
    d::NamedTuple;
    title="Residual correlation ρ12",
    kwargs...,
)
    ρ = Float64.(d.rho)
    fig = Figure(; size=(640, 360))
    if isempty(ρ)
        ax = Axis(fig[1, 1]; title=title, subtitle="no ρ12 (univariate / absent)")
        text!(ax, 0.5, 0.5; text="(no between-response correlation)", align=(:center, :center))
        hidedecorations!(ax)
        return fig
    end
    if d.constant
        ax = Axis(
            fig[1, 1];
            title=title,
            subtitle=@sprintf("constant ρ12 = %.4f (rho12 ~ 1)", ρ[1]),
            xlabel="ρ12",
            ylabel="",
        )
        ylims!(ax, 0.0, 1.0)
        xlims!(ax, -1.05, 1.05)
        vlines!(ax, [0.0]; color=(:gray, 0.4), linestyle=:dash)
        scatter!(
            ax,
            [ρ[1]],
            [0.5];
            marker=:circle,
            color=:white,
            strokecolor=:steelblue,
            strokewidth=2.0,
            markersize=16,
        )
    else
        n = length(ρ)
        ax = Axis(
            fig[1, 1];
            title=title,
            subtitle="observation-wise ρ12 = tanh(Xρ·β̂_ρ)",
            xlabel="observation index",
            ylabel="ρ12",
        )
        lines!(ax, 1:n, ρ; color=:steelblue, linewidth=1.5)
        scatter!(ax, 1:n, ρ; color=:steelblue, markersize=5)
        hlines!(ax, [0.0]; color=(:gray, 0.4), linestyle=:dash)
        ylims!(ax, -1.05, 1.05)
    end
    return fig
end

end # module
