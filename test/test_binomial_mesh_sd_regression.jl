using DRModels, Test, Random, LinearAlgebra, SparseArrays
import Distributions

@testset "Bernoulli mesh spatial-intercept SD regression" begin
    rng = MersenneTwister(20261027)
    G, m = 80, 24
    site_index = repeat(1:G; inner = m)
    site = ["site_$g" for g in site_index]
    z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = z_group[site_index]
    x = randn(rng, G * m)

    # Fixed sparse mesh precision and a two-vertex barycentric projection.
    Q = spdiagm(
        -1 => fill(-0.35, G - 1),
         0 => fill(2.0, G),
         1 => fill(-0.35, G - 1),
    )
    rows = collect(1:(G * m))
    vertices = site_index
    next_vertices = min.(vertices .+ 1, G)
    projection = sparse(
        vcat(rows, rows),
        vcat(vertices, next_vertices),
        vcat(fill(0.75, length(rows)), fill(0.25, length(rows))),
        G * m, G,
    )
    mesh = (; projection, precision = Q, alignment_ids = rows)

    alpha = [-0.55, 0.45]
    field = cholesky(Symmetric(Matrix(Q))).U \ randn(rng, G)
    projected = projection * field
    tau = exp.(alpha[1] .+ alpha[2] .* z_group)
    eta = 0.15 .+ 0.25 .* x .+ tau[site_index] .* projected
    y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(1 ./ (1 .+ exp.(-eta)))))
    data = (; y, x, z, site, row_id = rows)
    formula = bf(
        @formula(y ~ x + spatial(1 | site)),
        @formula(sd(site, spatial) ~ z),
    )

    fit = drm(formula, Binomial(); data, mesh, se = false, g_tol = 1e-5)

    @test fit.converged
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.35
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(G), z_group) * coef(fit, :sd))
    @test fit.scales[:sd] ≈ fit.scales[:sd_group][site_index]
    @test predict_parameters(fit, data; se = false)[:sd] ≈ fit.scales[:sd]
    simulator = DRModels._marginal_simulator_build(fit, data)
    @test simulator !== nothing
    draw_seed = 20261028
    actual = simulator(MersenneTwister(draw_seed))
    oracle_rng = MersenneTwister(draw_seed)
    latent = cholesky(Symmetric(Matrix(Q))).U \ randn(oracle_rng, G)
    eta_fixed = predict(fit, data; type = :link)
    p = 1 ./ (1 .+ exp.(-(eta_fixed .+ fit.scales[:sd] .* (projection * latent))))
    expected = Float64[rand(oracle_rng, Distributions.Binomial(
        1, clamp(p[i], eps(), 1 - eps()))) for i in eachindex(p)]
    @test actual == expected

    @test_throws ArgumentError drm(formula, Binomial(); data,
        mesh = (; projection = projection[:, 1:end-1], precision = Q), se = false)
    bad_precision = copy(Q); bad_precision[1, 1] = -1
    @test_throws ArgumentError drm(formula, Binomial(); data,
        mesh = (; projection, precision = bad_precision), se = false)
    reordered_data = merge(data, (; y = reverse(y), x = reverse(x),
        z = reverse(z), site = reverse(site), row_id = reverse(rows)))
    reordered_projection, _, reordered_ids =
        DRModels._binomial_mesh_inputs(mesh, reordered_data, :y)
    @test reordered_projection == projection[reverse(rows), :]
    @test reordered_ids == reverse(rows)
    bad_alignment = merge(data, (; row_id = ones(Int, length(rows))))
    @test_throws ArgumentError DRModels._binomial_mesh_inputs(mesh,
        bad_alignment, :y)
end
