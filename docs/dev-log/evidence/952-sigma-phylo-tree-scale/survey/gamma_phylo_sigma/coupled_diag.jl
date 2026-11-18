# Diagnose the Gamma coupled (1|p|phylo(species)) route on raw vs unit trees.
# Usage: julia --project=/opt/<base|fix> coupled_diag.jl <label>
using DRModels, DelimitedFiles, Printf, LinearAlgebra
const D = DRModels
const LABEL = isempty(ARGS) ? "unknown" : ARGS[1]
const N5 = "/tmp/claude-0/-home-user-DRModels-jl/bf8c2b27-03aa-5044-87f6-e06cfadb9c02/scratchpad/n5"

raw = readdlm(joinpath(N5, "carni70.csv"), ','; header = true)[1]
species = String.(strip.(String.(raw[:, 1]), '"'))
lsize = Float64.(raw[:, 2]); lrange = Float64.(raw[:, 3])
y = exp.(lsize); n = length(y)
nwk_raw = read(joinpath(N5, "carni70.nwk"), String)
nwk_unit = read(joinpath(N5, "carni70_unit.nwk"), String)
c = phylo_tree_height(augmented_phy(nwk_raw)) / phylo_tree_height(augmented_phy(nwk_unit))
println("# project=$LABEL c=$c"); flush(stdout)

Qr, gr, Gr = D._locscale_phylo_setup(nwk_raw, species)
Qu, gu, Gu = D._locscale_phylo_setup(nwk_unit, species)
println("# gidx equal=", gr == gu, " G=", Gr, " max|Qr*c - Qu|=", maximum(abs.(Qr .* c .- Qu)))
Zη = D._ls_canonical_Zeta(n); Zψ = D._ls_canonical_Zpsi(n)
kind = Val(:gamma)
Xμ = hcat(ones(n), lrange)
designs = [("sigma1", reshape(ones(n), n, 1)), ("sigmaX", hcat(ones(n), lrange))]

# engine packing: [βμ; βψ; logL11, L21, logL22]; raw = unit with Λ/c
function to_raw(θ, pμ, pψ)
    t = copy(θ); k = pμ + pψ
    t[k+1] -= 0.5log(c); t[k+2] /= sqrt(c); t[k+3] -= 0.5log(c)
    return t
end
ev(Q, G, gidx, Xψ, θ) = D._ls_whitened_eval(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ; gradient = false)
show_ev(tag, r) = @printf("EVAL project=%s %s ok=%s reason=%s value=%.17g\n", LABEL, tag,
                          r.status.ok, r.status.reason, r.status.ok ? r.value : NaN)

for (dn, Xψ) in designs
    pμ, pψ = size(Xμ, 2), size(Xψ, 2)
    β0 = D._ls_default_betastart(kind, y, Xμ)
    θ0 = vcat(β0, zeros(pψ), [log(0.3), 0.0, log(0.3)])
    show_ev("$dn start(log0.3) raw", ev(Qr, Gr, gr, Xψ, θ0))
    show_ev("$dn start(log0.3) unit", ev(Qu, Gu, gu, Xψ, θ0))
    show_ev("$dn unit-start-mapped-to-raw raw", ev(Qr, Gr, gr, Xψ, to_raw(θ0, pμ, pψ)))
    flush(stdout)
    t = time()
    fu = D._fit_locscale(kind, y, Xμ, Xψ, gu, Gu, Qu; g_tol = 1e-8, iterations = 2_000, se = false, whitened = true)
    @printf("FITENG project=%s %s unit default-start nll=%.17g loglik=%.10g converged=%s optimizer=%s time_s=%.1f theta=%s\n",
            LABEL, dn, fu.nll, -fu.nll, fu.converged, fu.optimizer, time() - t, string(round.(fu.θ; sigdigits = 10)))
    flush(stdout)
    θmap = to_raw(fu.θ, pμ, pψ)
    show_ev("$dn unit-optimum-mapped-to-raw raw", ev(Qr, Gr, gr, Xψ, θmap))
    flush(stdout)
    t = time()
    fr = D._fit_locscale(kind, y, Xμ, Xψ, gr, Gr, Qr; βμ0 = β0, βψ0 = zeros(pψ),
                         λ0 = [log(0.3) - 0.5log(c), 0.0, log(0.3) - 0.5log(c)],
                         g_tol = 1e-8, iterations = 2_000, se = false, whitened = true)
    @printf("FITENG project=%s %s raw start=log(0.3)-0.5log(c) nll=%.17g loglik=%.10g converged=%s optimizer=%s time_s=%.1f theta=%s |dll vs unit|=%.3e\n",
            LABEL, dn, fr.nll, -fr.nll, fr.converged, fr.optimizer, time() - t,
            string(round.(fr.θ; sigdigits = 10)), abs(fr.nll - fu.nll))
    flush(stdout)
end
println("# done $LABEL")
