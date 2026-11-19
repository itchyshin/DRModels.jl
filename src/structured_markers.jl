# structured_markers.jl — single source of truth for formula marker functions (#1375).
# `temporal` is a mean-only process marker (wave 1); the classic four feed
# phylogeny / pedigree / kinship / spatial structure.

const _STRUCTURED_MARKERS = (relmat, animal, phylo, spatial)
const _STRUCTURED_OR_TEMPORAL_MARKERS = (_STRUCTURED_MARKERS..., temporal)

# Kind symbol for a marker `FunctionTerm`, or `nothing`.
function _structured_marker_kind_symbol(t::FunctionTerm)
    f = t.f
    f === relmat   && return :relmat
    f === animal   && return :animal
    f === phylo    && return :phylo
    f === spatial  && return :spatial
    f === temporal && return :temporal
    return nothing
end

# Kind symbol or error (for q=4 splits that must see a marker).
function _structured_marker_kind(t::FunctionTerm)
    kind = _structured_marker_kind_symbol(t)
    kind === nothing && error("unsupported structured marker")
    return kind
end

_is_structured_marker(t::FunctionTerm) = t.f in _STRUCTURED_MARKERS
_is_structured_or_temporal_marker(t::FunctionTerm) = t.f in _STRUCTURED_OR_TEMPORAL_MARKERS

const _STRUCTURED_MARKER_NAMES = (
    relmat => "relmat",
    animal => "animal",
    phylo  => "phylo",
    spatial => "spatial",
    temporal => "temporal",
)