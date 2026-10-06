# test_parity_biv_q4_phylo_reml.jl — standalone same-target fixture for
# biv_q4_phylo_reml (issue #433).
#
# Wired into test/runtests.jl by Option A (#445) after #423+#428 landed.
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_parity_biv_q4_phylo_reml.jl")'
#
# Claim fence: this file checks a native-vs-Julia same-target cell within the
# row's declared [tol]. It does not claim R–Julia parity complete, interval
# coverage/reliability, AI-REML, or R-via-Julia bridge admission.
# Workflow G fixtures stay 0.6.0 / ML / no tree; this cell is 0.7.0 REML + tree
# outside test/parity/fixtures/.

# #476: wrapped in a module so this file's FIXTURE/_load_data/_coef_named/
# _within cannot collide with the same names in another parity test file
# included later by runtests.jl (they used to live bare in Main).
module TestParityBivQ4PhyloREML

using DRModels
using Test
using TOML
using DelimitedFiles: readdlm
const FIXTURE = joinpath(@__DIR__, "parity", "q4-reml", "biv-q4-phylo-reml")

function _load_data(dir)
    raw, header = readdlm(joinpath(dir, "data.csv"), ','; header = true)
    cols = Symbol.(strip.(string.(vec(header))))
    numeric = Set((:y1, :y2, :x))
    pairs = map(enumerate(cols)) do (j, name)
        col = raw[:, j]
        if name in numeric
            name => Float64[parse(Float64, string(v)) for v in col]
        else
            name => string.(col)
        end
    end
    return NamedTuple(pairs)
end

function _coef_named(fit)
    θ = coef(fit)
    namemap = Dict(p => ns for (p, ns) in fit.coefnames)
    out = Dict{String,Float64}()
    for (param, r) in fit.blocks
        param === :phylocov && continue
        haskey(namemap, param) || continue
        names = namemap[param]
        slice = θ[r]
        length(names) == length(slice) || error("name/coef mismatch for $param")
        for (nm, est) in zip(names, slice)
            out["$(param)_$(nm)"] = est
        end
    end
    return out
end

_within(a, b, rtol, atol) = abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))

@testset "biv_q4_phylo_reml same-target fixture (#433)" begin
    @test isfile(joinpath(FIXTURE, "data.csv"))
    @test isfile(joinpath(FIXTURE, "tree.newick"))
    @test isfile(joinpath(FIXTURE, "expected.toml"))
    @test isfile(joinpath(FIXTURE, "expected.meta.toml"))

    expected = TOML.parsefile(joinpath(FIXTURE, "expected.toml"))
    meta = TOML.parsefile(joinpath(FIXTURE, "expected.meta.toml"))
    @test meta["drmtmb_version"] == "0.7.0"
    @test haskey(meta, "r_call")
    @test haskey(meta, "seed")
    @test expected["fit"]["method"] == "REML"
    @test expected["fit"]["engine"] == "tmb"
    @test haskey(expected, "status")
    @test haskey(expected["status"], "converged")
    @test haskey(expected["status"], "pdHess")
    @test haskey(expected["status"], "interval_status")
    # Status is recorded; this is not a coverage / reliability claim.
    @test expected["status"]["interval_status"] != "coverage_claimed"

    dat = _load_data(FIXTURE)
    tree = read(joinpath(FIXTURE, "tree.newick"), String)
    form = bf(mu1    = @formula(y1 ~ x + phylo(1 | species)),
              mu2    = @formula(y2 ~ x + phylo(1 | species)),
              sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
              sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
              rho12  = @formula(rho12 ~ 1))
    fit = drm(form, Gaussian(); data = dat, tree = tree, method = :REML, q4_vcov = false)

    @test estimation_method(fit) === :REML
    @test isfinite(reml_loglik(fit))
    @test isfinite(loglik(fit))
    # #484 landed the automatic warm restart that lets drm()'s public REML
    # route converge on this cell; the follow-up fixture re-derivation (same
    # day) re-measured `status.julia_converged` so the TOML field is now an
    # accurate statement about the public route. Read it directly, like every
    # other [status] field checked above, instead of pinning `true` inline.
    # See test/test_q4_reml_warm_restart.jl for the engine-level
    # g_residual < g_tol check.
    @test is_converged(fit) == expected["status"]["julia_converged"]

    atol_ll = Float64(get(expected["tol"], "atol_loglik", 1e-3))
    atol_c = Float64(get(expected["tol"], "atol_coef", 1e-3))
    rtol_c = Float64(get(expected["tol"], "rtol_coef", 1e-3))
    ref_ll = Float64(expected["fit"]["loglik"])
    @test _within(loglik(fit), ref_ll, 0.0, atol_ll)

    got = _coef_named(fit)
    for (name, ref) in expected["coef"]
        @test haskey(got, name)
        @test _within(got[name], Float64(ref), rtol_c, atol_c)
    end
end

end # module TestParityBivQ4PhyloREML
