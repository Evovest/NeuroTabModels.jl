"""
    _CategoricalEmbeddings(cardinalities, d_embedding)

Embeds categorical level codes. Input `(ncat, batch)` holds codes `0:K_j` for feature `j`,
`0` marking a level unseen at fit time. All features share one table, feature `j` owning
`K_j + 1` consecutive rows, so a whole batch is a single lookup.
Output shape `(d_embedding, ncat, batch)`.

# Arguments
- `cardinalities::AbstractVector{<:Integer}`: Number of training levels `K_j` per feature.
- `d_embedding::Int`: Embedding dimension per feature.
"""
struct _CategoricalEmbeddings{E} <: LuxCore.AbstractLuxContainerLayer{(:table,)}
    table::E
    offsets::Vector{Int32}
end

function _CategoricalEmbeddings(cardinalities::AbstractVector{<:Integer}, d_embedding::Int)
    rows = cardinalities .+ 1
    offsets = Int32.(cumsum(rows) .- rows .+ 1)
    return _CategoricalEmbeddings(Embedding(sum(rows) => d_embedding; init_weight=_embed_affine_init), offsets)
end

# `offsets` lives in `st` so that `dev(st)` moves it to the device with the rest of the state.
function LuxCore.initialstates(rng::AbstractRNG, l::_CategoricalEmbeddings)
    return (table=LuxCore.initialstates(rng, l.table), offsets=l.offsets)
end

function (l::_CategoricalEmbeddings)(x::AbstractMatrix, ps, st)
    idx = unsafe_trunc.(Int32, x) .+ st.offsets
    y, st_t = l.table(idx, ps.table, st.table)
    return y, (table=st_t, offsets=st.offsets)
end
