_untuple(x) = x isa Tuple ? x[1] : x

"""
    ResidualScale(init)

Learned scalar on a residual branch. One parameter, initialized at `init`
(default `0.1`) so peer attention starts as a small add-on rather than a
full-scale mix of the batch / group.
"""
struct ResidualScale <: LuxCore.AbstractLuxLayer
    init::Float32
end

function LuxCore.initialparameters(::AbstractRNG, l::ResidualScale)
    return (; s=Float32[l.init])
end
LuxCore.parameterlength(::ResidualScale) = 1
LuxCore.statelength(::ResidualScale) = 0

(l::ResidualScale)(x::AbstractArray, ps, st) = x .* ps.s, st

"""
    AttnResidual(hsize, nheads; dropout=0.0, attn_dropout=0.0, attn_scale=0.1f0)

Peer attention over an unordered set (attention batch dim = 1). Shared ``W_{qk}``
for query and key, values = encoder tokens. Score scale is the usual
`NNlib.dot_product_attention` ``1/√d``. Residual
`x + scale * Dropout(Attn)`, with `scale` a learned scalar initialized at
`attn_scale` (default `0.1`). Encoder tokens should already be BatchNorm'd so
QK logits stay O(1); the small residual scale is what lets the model ignore
peers when the batch is not a real group.

Inputs are `(hidden, seq)` feature-first matrices. An optional key-padding mask
may be passed as `(x, mask)` so padded group-buffer slots are ignored.
"""
struct AttnResidual{QK,AD,D,S} <: LuxCore.AbstractLuxContainerLayer{(:qk_proj, :attn_drop, :drop, :scale)}
    qk_proj::QK
    attn_drop::AD
    drop::D
    scale::S
    nheads::Int
end

function AttnResidual(
    hsize::Int,
    nheads::Int;
    dropout::Float64=0.0,
    attn_dropout::Float64=0.0,
    attn_scale::Float32=0.1f0,
    identity_qk::Bool=false,
)
    qk = identity_qk ? NoOpLayer() : Dense(hsize => hsize; use_bias=false)
    return AttnResidual(qk, Dropout(attn_dropout), Dropout(dropout), ResidualScale(attn_scale), nheads)
end

_as_seq(x::AbstractMatrix) = reshape(x, size(x, 1), size(x, 2), 1)
_from_seq(x::AbstractArray) = reshape(x, size(x, 1), size(x, 2))

# Per-token RMS over the hidden axis (MLP QK). Per-channel RMS over the sequence
# (NeuroTree: each ensemble is its own head, so do not mix `k` in the denominator).
function _rms_tokens(x::AbstractMatrix)
    T = eltype(x)
    return x ./ sqrt.(sum(abs2, x; dims=1) ./ T(size(x, 1)) .+ T(1.0f-5))
end
function _rms_channels(x::AbstractMatrix, mask=nothing)
    T = eltype(x)
    seq = size(x, 2)
    if mask === nothing
        return x ./ sqrt.(sum(abs2, x; dims=2) ./ T(seq) .+ T(1.0f-5))
    end
    w = reshape(T.(vec(view(mask, :, 1, 1, 1))), 1, seq)
    n = max(sum(w), one(T))
    ss = sum(abs2.(x) .* w; dims=2)
    return x ./ sqrt.(ss ./ n .+ T(1.0f-5))
end

function _block(l::AttnResidual, x::AbstractMatrix, mask, ps, st)
    qk, st_qk = if l.qk_proj isa NoOpLayer
        _rms_channels(x, mask), st.qk_proj
    else
        l.qk_proj(_rms_tokens(x), ps.qk_proj, st.qk_proj)
    end
    attn_drop = StatefulLuxLayer(l.attn_drop, ps.attn_drop, st.attn_drop)
    a3, _ = dot_product_attention(
        _as_seq(qk), _as_seq(qk), _as_seq(x); nheads=l.nheads, mask, fdrop=attn_drop
    )
    a, st_d = l.drop(_from_seq(a3), ps.drop, st.drop)
    s, st_s = l.scale(a, ps.scale, st.scale)
    return x .+ s, (; qk_proj=st_qk, attn_drop=attn_drop.st, drop=st_d, scale=st_s)
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

function _attn_blocks(
    hsize::Int,
    nheads::Int,
    n_attn_layers::Int,
    dropout::Float64,
    attn_dropout::Float64,
    attn_scale::Float32=0.1f0,
)
    n_attn_layers >= 0 || error("`n_attn_layers` must be ≥ 0, got $n_attn_layers.")
    n_attn_layers == 0 && return NoOpLayer()
    blocks = [AttnResidual(hsize, nheads; dropout, attn_dropout, attn_scale) for _ in 1:n_attn_layers]
    return Chain(blocks...)
end

# NeuroTree tokens *are* the k ensemble predictions. A shared Dense QK would mix
# those ensembles; encoder `dropout` on the residual would drop predictions.
# One head per channel (`nheads = hsize`), identity QK, no residual dropout.
function _tree_attn_blocks(hsize::Int, n_attn_layers::Int, attn_dropout::Float64, attn_scale::Float32)
    n_attn_layers >= 0 || error("`n_attn_layers` must be ≥ 0, got $n_attn_layers.")
    n_attn_layers == 0 && return NoOpLayer()
    blocks = [
        AttnResidual(hsize, hsize; dropout=0.0, attn_dropout, attn_scale, identity_qk=true) for _ in 1:n_attn_layers
    ]
    return Chain(blocks...)
end

_pred_head(hsize::Int, outsize::Int) = Dense(hsize => outsize)
