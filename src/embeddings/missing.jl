"""
    _MissingTokens(embedding, d_embedding, nfeats)

Embeds missing inputs (`NaN`) as a learned per-feature `[MISS]` vector, and observed inputs
with `embedding`.
The embedding sees `0` in place of a missing value, and its output there is replaced by that
feature's `[MISS]` vector. All features' vectors start from one shared vector.
Produces a `(d_embedding, nfeats, batch)` tensor.

# Arguments
- `embedding`: Per-feature embedding layer with `(d_embedding, nfeats, batch)` output.
- `d_embedding::Int`: Embedding dimension per feature.
- `nfeats::Int`: Number of input features.
"""
struct _MissingTokens{E} <: LuxCore.AbstractLuxContainerLayer{(:embedding,)}
    embedding::E
    d_embedding::Int
    nfeats::Int
end

function LuxCore.initialparameters(rng::AbstractRNG, l::_MissingTokens)
    token = _embed_affine_init(rng, l.d_embedding)
    return (; embedding=LuxCore.initialparameters(rng, l.embedding), token=repeat(token, 1, l.nfeats))
end

_missing_flag(x) = ifelse(isnan(x), one(x), zero(x))
_observed(x) = ifelse(isnan(x), zero(x), x)

function (l::_MissingTokens)(x::AbstractMatrix, ps, st)
    miss = _as_feature_groups(_missing_flag.(x))
    e, st_e = l.embedding(_observed.(x), ps.embedding, st.embedding)
    return e .+ miss .* (ps.token .- e), (; embedding=st_e)
end
