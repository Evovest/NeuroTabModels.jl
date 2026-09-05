_untuple(x) = x isa Tuple ? x[1] : x

"""
    AttnResidual(hsize, nheads; dropout=0.0, attn_dropout=0.0)

Peer attention over an unordered set (attention batch dim = 1). Shared ``W_{qk}``
for query and key, values = encoder tokens. Scale is the usual
`NNlib.dot_product_attention` ``1/√d``. Residual `x + Dropout(Attn)`.

Inputs are `(hidden, seq)` feature-first matrices. An optional key-padding mask
may be passed as `(x, mask)` so padded group-buffer slots are ignored.
"""
struct AttnResidual{QK,AD,D} <: LuxCore.AbstractLuxContainerLayer{(:qk_proj, :attn_drop, :drop)}
    qk_proj::QK
    attn_drop::AD
    drop::D
    nheads::Int
end

function AttnResidual(hsize::Int, nheads::Int; dropout::Float64=0.0, attn_dropout::Float64=0.0)
    return AttnResidual(
        Dense(hsize => hsize; use_bias=false),
        Dropout(attn_dropout),
        Dropout(dropout),
        nheads,
    )
end

_as_seq(x::AbstractMatrix) = reshape(x, size(x, 1), size(x, 2), 1)
_from_seq(x::AbstractArray) = reshape(x, size(x, 1), size(x, 2))

function _block(l::AttnResidual, x::AbstractMatrix, mask, ps, st)
    qk, st_qk = l.qk_proj(x, ps.qk_proj, st.qk_proj)
    attn_drop = StatefulLuxLayer(l.attn_drop, ps.attn_drop, st.attn_drop)
    a3, _ = dot_product_attention(
        _as_seq(qk), _as_seq(qk), _as_seq(x); nheads=l.nheads, mask, fdrop=attn_drop
    )
    a, st_d = l.drop(_from_seq(a3), ps.drop, st.drop)
    return x .+ a, (; qk_proj=st_qk, attn_drop=attn_drop.st, drop=st_d)
end

(l::AttnResidual)(x::AbstractMatrix, ps, st) = _block(l, x, nothing, ps, st)
function (l::AttnResidual)((x, mask)::Tuple, ps, st)
    y, st_ = _block(l, x, mask, ps, st)
    return (y, mask), st_
end

"""
    _key_padding_mask(w, seq)

Build a boolean key-padding mask for `NNlib.dot_product_attention`.

Attention scores have shape `(kv_len, q_len, nheads, batch)`. A `true` entry keeps that
key position. Reshaping valid-token flags to `(seq, 1, 1, 1)` zeros out entire *columns*
of keys (padded buffer slots) for every query — a rectangular mask, not a causal triangle.
"""
_valid_tokens(v::AbstractVector{Bool}) = v
_valid_tokens(v::AbstractVector) = v .> zero(eltype(v))

function _key_padding_mask(w, seq::Int)
    return reshape(_valid_tokens(vec(w)), seq, 1, 1, 1)
end

function _attn_blocks(hsize::Int, nheads::Int, n_attn_layers::Int, dropout::Float64, attn_dropout::Float64)
    n_attn_layers >= 0 || error("`n_attn_layers` must be ≥ 0, got $n_attn_layers.")
    n_attn_layers == 0 && return NoOpLayer()
    blocks = [AttnResidual(hsize, nheads; dropout, attn_dropout) for _ in 1:n_attn_layers]
    return Chain(blocks...)
end

_pred_head(hsize::Int, outsize::Int) = Dense(hsize => outsize)
