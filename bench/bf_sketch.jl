################################################################################
# bf_sketch.jl — brms-style bf() multi-formula parser for DRModels.jl
#
# This is a SKETCH demonstrating how
#
#     bf(mu    = @formula(y ~ x + (1 | p | id)),
#        sigma = @formula(_ ~ x + (1 | p | id)))
#
# would work in Julia, including the `(1 | p | id)` shared-label random-
# effect trick from brms / drmTMB. The middle pipe `p` is a LABEL that
# ties matching RE terms across dpars into one structured covariance
# block.
#
# Not yet integrated with the POC — read end-to-end as a design artifact.
# The StatsModels.jl extension points are real; the design-matrix
# materialization stubs out at the boundary with MixedModels.jl.
#
# Structure:
#   1. Term types (SimpleRE, SharedRE, PhyloRE) + DistributionalFormula
#   2. @formula extension to recognize `(rhs | label | group)`
#   3. bf() constructor
#   4. Cross-dpar label-matching: collect RE terms by label into blocks
#   5. Covariance-block parameter layout (log-Cholesky)
#   6. Worked examples at the bottom
################################################################################

module BFSketch

using StatsModels: AbstractTerm, ConstantTerm, FormulaTerm, FunctionTerm,
                   InteractionTerm, Schema, Term, apply_schema, term

# Stub model type — DRModels.jl would supply the real one.
abstract type DistributionalRegressionModel end


# =============================================================================
# 1. TERM TYPES
# =============================================================================

# A plain random-effects term — the lme4 `(rhs | group)` form, no label.
# Equivalent to MixedModels.jl's RandomEffectsTerm; we keep our own thin
# wrapper because we also need the `dpar` field.
struct SimpleRE
    rhs::Any                  # left of |, e.g. ConstantTerm(1) or `1 + x`
    group::Symbol             # grouping factor, e.g. :id
    dpar::Symbol              # which submodel — :mu, :sigma, :mu1, ...
end

# A random-effects term carrying a shared-block LABEL.
# Produced when the parser sees the three-pipe `(rhs | label | group)`
# form. Terms with the same label across different dpars get assembled
# into one structured covariance block at bf()-time.
struct SharedRE
    rhs::Any                  # left of first |
    label::Symbol             # middle pipe — e.g. :p
    group::Symbol             # grouping factor — e.g. :id
    dpar::Symbol              # filled in by bf()
end

# A phylogenetic random-effects term — `phylo(rhs | label? | group)`.
# Carries the (optional) shared-block label exactly like SharedRE. The
# tree itself is passed to fit_drm() as a kwarg and resolved later by
# matching on `group`.
struct PhyloRE
    rhs::Any
    label::Union{Symbol, Nothing}
    group::Symbol             # the species-id factor
    dpar::Symbol
end

# Container produced by bf() — one formula per distributional parameter
# plus cross-dpar block bookkeeping.
struct DistributionalFormula
    formulas::Dict{Symbol, FormulaTerm}        # :mu => @formula(y ~ x), etc.
    shared_blocks::Dict{Symbol, Vector{SharedRE}}    # label => terms
    phylo_blocks::Dict{Symbol, Vector{PhyloRE}}      # label => terms
    simple_res::Vector{SimpleRE}               # no-label REs
    dpars::Vector{Symbol}
    n_response::Int                            # 1 or 2
end


# =============================================================================
# 2. FORMULA-LEVEL PARSING — recognize `(rhs | label | group)`
# =============================================================================
#
# Julia parses `1 | p | id` as `((1 | p) | id)` — left-associative chained |.
# Inside @formula this surfaces as nested FunctionTerm{typeof(|)} nodes.
# We hook into StatsModels' apply_schema to recognize the nested form.
#
# Precedent: MixedModels.jl uses the same mechanism for `zerocorr(1 + x | g)`
# and for the standard `(1 | g)` form.

# Stub function so @formula(... + phylo(1 | p | species) ... ) parses.
# Inside formulas it's a marker; outside it should never be called.
phylo() = error("phylo() may only appear inside @formula(...)")

# Dispatch on the outer `|` FunctionTerm. Two cases:
#
#   (a) outer.args[1] is itself a FunctionTerm{|}  → THREE-pipe form → SharedRE
#   (b) otherwise                                   → standard (rhs | group) → SimpleRE
#
function apply_schema(
    t::FunctionTerm{typeof(|)},
    schema::Schema,
    Mod::Type{<:DistributionalRegressionModel},
)
    lhs, rhs = t.args[1], t.args[2]

    if lhs isa FunctionTerm{typeof(|)}
        # Three-pipe form: `(rhs | label | group)` parsed as `((rhs | label) | group)`
        inner_rhs    = lhs.args[1]                 # e.g. ConstantTerm(1) or `1 + x`
        label_symbol = _extract_symbol(lhs.args[2])
        group_symbol = _extract_symbol(rhs)
        return SharedRE(inner_rhs, label_symbol, group_symbol, :__UNFILLED__)
    end

    # Two-pipe form: `(rhs | group)`
    return SimpleRE(lhs, _extract_symbol(rhs), :__UNFILLED__)
end

# Dispatch on `phylo(...)` FunctionTerm. The argument is one of the |-forms.
function apply_schema(
    t::FunctionTerm{typeof(phylo)},
    schema::Schema,
    Mod::Type{<:DistributionalRegressionModel},
)
    arg = first(t.args)
    arg isa FunctionTerm{typeof(|)} ||
        error("phylo() expects a RE term inside, got $(typeof(arg))")

    inner = apply_schema(arg, schema, Mod)
    return if inner isa SharedRE
        PhyloRE(inner.rhs, inner.label, inner.group, :__UNFILLED__)
    elseif inner isa SimpleRE
        PhyloRE(inner.rhs, nothing, inner.group, :__UNFILLED__)
    else
        error("phylo() inner term unexpected: $(typeof(inner))")
    end
end

_extract_symbol(t::Term)   = t.sym
_extract_symbol(s::Symbol) = s
_extract_symbol(t)         = error("expected a Symbol-valued term, got $(typeof(t))")


# =============================================================================
# 3. THE bf() CONSTRUCTOR
# =============================================================================
#
# Mirrors brms::bf(). A single positional formula defaults to the :mu
# submodel; keyword args supply other dpars. The convention for non-
# response dpars (sigma, rho12, nu, ...) is to write `_ ~ x1 + x2` — the
# underscore signals "no response on the LHS, this is a submodel".

function bf(args...; kwargs...)
    formulas = Dict{Symbol, FormulaTerm}()

    if length(args) == 1
        formulas[:mu] = args[1]
    elseif length(args) > 1
        error("bf() takes at most one positional formula (the :mu submodel)")
    end

    for (dpar, f) in kwargs
        haskey(formulas, dpar) && error("dpar :$dpar specified twice")
        formulas[dpar] = f
    end

    isempty(formulas) && error("bf() requires at least one formula")

    dpars = collect(keys(formulas))
    n_response = _count_responses(dpars)

    return _build_distributional_formula(formulas, dpars, n_response)
end

function _count_responses(dpars::Vector{Symbol})
    biv = any(d -> d in (:mu1, :mu2, :sigma1, :sigma2, :rho12), dpars)
    uni = any(d -> d in (:mu, :sigma, :nu, :zi), dpars)
    biv && uni && error("bf() mixes univariate and bivariate dpars — pick one set")
    biv && return 2
    uni && return 1
    error("bf() must include :mu (univariate) or :mu1/:mu2 (bivariate)")
end


# =============================================================================
# 4. BLOCK ASSEMBLY — collect RE terms by label across dpars
# =============================================================================

function _build_distributional_formula(
    formulas::Dict{Symbol, FormulaTerm},
    dpars::Vector{Symbol},
    n_response::Int,
)
    shared_blocks = Dict{Symbol, Vector{SharedRE}}()
    phylo_blocks  = Dict{Symbol, Vector{PhyloRE}}()
    simple_res    = SimpleRE[]

    for dpar in dpars
        for term in _walk_rhs(formulas[dpar].rhs)
            if term isa SharedRE
                t = SharedRE(term.rhs, term.label, term.group, dpar)
                push!(get!(shared_blocks, t.label, SharedRE[]), t)

            elseif term isa PhyloRE
                t = PhyloRE(term.rhs, term.label, term.group, dpar)
                # Unlabeled phylo terms get a synthetic per-dpar label so
                # they don't accidentally share a block with anything.
                key = something(t.label, Symbol("__phylo_unlabeled_$(dpar)"))
                push!(get!(phylo_blocks, key, PhyloRE[]), t)

            elseif term isa SimpleRE
                push!(simple_res, SimpleRE(term.rhs, term.group, dpar))
            end
        end
    end

    _validate_shared_blocks(shared_blocks)
    _validate_phylo_blocks(phylo_blocks)

    return DistributionalFormula(
        formulas, shared_blocks, phylo_blocks, simple_res,
        dpars, n_response,
    )
end

# Walk the RHS of a formula, returning a flat vector of leaf terms.
_walk_rhs(t::Tuple) = vcat(map(_walk_rhs, t)...)
_walk_rhs(t::AbstractTerm) = [t]
_walk_rhs(t::SharedRE) = [t]
_walk_rhs(t::PhyloRE)  = [t]
_walk_rhs(t::SimpleRE) = [t]

# Within a single label, every term must share the same group factor.
# Otherwise the "shared block" idea doesn't make sense.
function _validate_shared_blocks(blocks::Dict{Symbol, Vector{SharedRE}})
    for (label, terms) in blocks
        groups = unique(t.group for t in terms)
        length(groups) == 1 || error(
            "Shared-RE label :$label appears with multiple grouping factors $groups; " *
            "all terms with the same label must share one grouping factor."
        )
        dpars = [t.dpar for t in terms]
        length(unique(dpars)) == length(dpars) || error(
            "Shared-RE label :$label appears more than once for the same dpar; " *
            "labels are meant to tie ACROSS dpars, not within."
        )
    end
end

_validate_phylo_blocks(blocks::Dict{Symbol, Vector{PhyloRE}}) =
    _validate_shared_blocks(
        Dict(k => [SharedRE(t.rhs, k, t.group, t.dpar) for t in v]
             for (k, v) in blocks)
    )


# =============================================================================
# 5. COVARIANCE-BLOCK PARAMETER LAYOUT
# =============================================================================
#
# For each label with k associated terms (across k dpars), parameterize
# a k×k positive-definite covariance via log-Cholesky:
#
#   log_sd::Vector{T}        of length k        — log marginal SDs
#   chol_offdiag::Vector{T}  of length k(k-1)/2 — strict-lower of L
#
# where L_corr is built so that diag(L_corr L_corr') == I (row-normalized
# unconstrained Cholesky factor). Then
#
#   Σ = diag(exp(log_sd)) · (L_corr L_corr') · diag(exp(log_sd))
#
# Total params per block: k + k(k-1)/2.
#
# Examples:
#   k=2 (mu + sigma share id):           2 + 1 = 3 params
#   k=4 (q=4 phylo, mu1/mu2/sigma1/sigma2): 4 + 6 = 10 params
#
# This is the same parameterization MixedModels.jl uses internally and
# what drmTMB uses in C++ via TMB::density::UNSTRUCTURED_CORR_t.

struct CovBlockSpec
    label::Symbol
    kind::Symbol               # :shared or :phylo
    dpars::Vector{Symbol}      # which submodels participate (length k)
    group::Symbol              # grouping factor (e.g. :id, :species)
    k::Int                     # block size
    n_par::Int                 # k + k(k-1)/2
end

function block_specs(df::DistributionalFormula)
    out = CovBlockSpec[]

    for (label, terms) in df.shared_blocks
        k = length(terms)
        push!(out, CovBlockSpec(
            label, :shared, [t.dpar for t in terms], terms[1].group, k,
            k + k * (k - 1) ÷ 2,
        ))
    end

    for (label, terms) in df.phylo_blocks
        k = length(terms)
        push!(out, CovBlockSpec(
            label, :phylo, [t.dpar for t in terms], terms[1].group, k,
            k + k * (k - 1) ÷ 2,
        ))
    end

    return out
end


# =============================================================================
# 6. WORKED EXAMPLES (what bf() produces; uncomment to demo)
# =============================================================================
#
# # Example A — univariate, correlated id-level RE on mu and sigma
# #
# # User writes:
# #     f = bf(mu    = @formula(y ~ x + (1 | p | id)),
# #            sigma = @formula(_ ~ x + (1 | p | id)))
# #
# # Result:
# #     f.dpars         == [:mu, :sigma]
# #     f.n_response    == 1
# #     f.shared_blocks == Dict(:p => [
# #         SharedRE(ConstantTerm(1), :p, :id, :mu),
# #         SharedRE(ConstantTerm(1), :p, :id, :sigma)
# #     ])
# #     block_specs(f)  == [
# #         CovBlockSpec(:p, :shared, [:mu, :sigma], :id, 2, 3)
# #     ]
# #     # → 3 params per id: log_sd_mu, log_sd_sigma, atanh_corr
#
# # Example B — bivariate q=4 phylogenetic location-scale block
# #
# # User writes:
# #     f = bf(mu1    = @formula(y1 ~ x + phylo(1 | core | species)),
# #            mu2    = @formula(y2 ~ x + phylo(1 | core | species)),
# #            sigma1 = @formula(_  ~ x + phylo(1 | core | species)),
# #            sigma2 = @formula(_  ~ x + phylo(1 | core | species)),
# #            rho12  = @formula(_  ~ 1))
# #
# # Result:
# #     f.dpars       == [:mu1, :mu2, :sigma1, :sigma2, :rho12]
# #     f.n_response  == 2
# #     f.phylo_blocks == Dict(:core => [4 PhyloREs, one per of mu1/mu2/sigma1/sigma2])
# #     block_specs(f) == [
# #         CovBlockSpec(:core, :phylo, [:mu1, :mu2, :sigma1, :sigma2],
# #                       :species, 4, 10)
# #     ]
# #     # → 10 params: 4 log_sd + 6 atanh_corr in the q=4 block
# #     # → matches drmTMB's `density::UNSTRUCTURED_CORR_t` layout exactly
#
# # Example C — block-diagonal: separate q=2 location + q=2 scale (brms idiom)
# #
# # User writes:
# #     f = bf(mu1    = @formula(y1 ~ phylo(1 | pl | species)),
# #            mu2    = @formula(y2 ~ phylo(1 | pl | species)),
# #            sigma1 = @formula(_  ~ phylo(1 | ps | species)),
# #            sigma2 = @formula(_  ~ phylo(1 | ps | species)),
# #            rho12  = @formula(_  ~ 1))
# #
# # Result:
# #     f.phylo_blocks == Dict(:pl => [2 PhyloREs], :ps => [2 PhyloREs])
# #     block_specs(f) == [
# #         CovBlockSpec(:pl, :phylo, [:mu1, :mu2],       :species, 2, 3),
# #         CovBlockSpec(:ps, :phylo, [:sigma1, :sigma2], :species, 2, 3)
# #     ]
# #     # → block-diagonal: 6 total params instead of the 10 of the q=4 form
#
# # Example D — plain lme4-style (no label, no block sharing)
# #
# # User writes:
# #     f = bf(mu    = @formula(y ~ x + (1 | id) + (1 + x | site)),
# #            sigma = @formula(_ ~ x))
# #
# # Result:
# #     f.simple_res == [
# #         SimpleRE(ConstantTerm(1),       :id,   :mu),
# #         SimpleRE(InterceptAndSlope(:x), :site, :mu)
# #     ]
# #     f.shared_blocks == Dict()
# #     # → standard MixedModels.jl-style fit, no cross-dpar correlation


# =============================================================================
# OPEN HOOKS (delegated, not in this sketch)
# =============================================================================
#
# After bf() returns a DistributionalFormula, the fit pipeline needs:
#
#   a) Materialize per-dpar fixed-effect design matrices X_mu, X_sigma, ...
#      from each formula's RHS minus the RE terms.
#      → MixedModels.jl-style: strip RE terms, call modelmatrix(...) on the rest.
#      → ~50 LOC.
#
#   b) Materialize per-block RE design matrices Z_block from grouping factors.
#      → Sparse incidence matrix per (term, group). Standard pattern.
#      → ~80 LOC.
#
#   c) Build the parameter packing vector laid out as:
#          [β per dpar...; log_sd per block...; chol_offdiag per block...]
#      → Provides `pack(θ_full) -> θ_flat` and `unpack(θ_flat) -> θ_full`.
#      → ~100 LOC; mirrors GLLVM.jl's `src/packing.jl`.
#
#   d) Likelihood: for Gaussian families, integrate out the REs analytically
#      via marginal covariance (closed-form). For non-Gaussian, hand off
#      to a Laplace wrapper (deferred to DRModels.jl v0.3+).
#      → For v0.1.x: Gaussian only. ~150 LOC reusing GLLVM.jl block patterns.

end  # module BFSketch
