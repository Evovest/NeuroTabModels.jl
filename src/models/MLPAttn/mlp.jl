module MLPAttn

export MLPAttnConfig

using Lux
using LuxCore
using Random: AbstractRNG

import ..Models: Architecture, get_activation, uses_batch_mask

"""
    ResidualScale(init)

Single learned scalar on a residual branch. One parameter, so peer attention starts
as a small add-on (`init`, default `0.1`) rather than a full-scale mix.
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

Peer-context residual: `x + scale * Dropout(MHA(x))`.

No extra LayerNorm and no FFN — the item encoder already has capacity and (for MLPAttn)
BatchNorm. One learned scalar `scale` keeps the mix small at init.

Inputs are `(hidden, seq)` feature-first matrices. An optional key-padding mask
may be passed as `(x, mask)` so padded group-buffer slots are ignored.
"""
struct AttnResidual{A,S,D} <: LuxCore.AbstractLuxContainerLayer{(:mha, :scale, :drop)}
    mha::A
    scale::S
    drop::D
end

function AttnResidual(
    hsize::Int, nheads::Int; dropout::Float64=0.0, attn_dropout::Float64=0.0, attn_scale::Float32=0.1f0
)
    return AttnResidual(
        MultiHeadAttention(hsize; nheads, attention_dropout_probability=Float32(attn_dropout)),
        ResidualScale(attn_scale),
        Dropout(dropout),
    )
end

_as_seq(x::AbstractMatrix) = reshape(x, size(x, 1), size(x, 2), 1)
_from_seq(x::AbstractArray) = reshape(x, size(x, 1), size(x, 2))

function _mha(mha, x::AbstractMatrix, ::Nothing, ps, st)
    (y, _), st_ = mha(_as_seq(x), ps, st)
    return _from_seq(y), st_
end
function _mha(mha, x::AbstractMatrix, mask, ps, st)
    x3 = _as_seq(x)
    (y, _), st_ = mha((x3, x3, x3, mask), ps, st)
    return _from_seq(y), st_
end

function _block(l::AttnResidual, x::AbstractMatrix, mask, ps, st)
    a, st_a = _mha(l.mha, x, mask, ps.mha, st.mha)
    a, st_s = l.scale(a, ps.scale, st.scale)
    a, st_d = l.drop(a, ps.drop, st.drop)
    return x .+ a, (; mha=st_a, scale=st_s, drop=st_d)
end

(l::AttnResidual)(x::AbstractMatrix, ps, st) = _block(l, x, nothing, ps, st)
function (l::AttnResidual)((x, mask)::Tuple, ps, st)
    y, st_ = _block(l, x, mask, ps, st)
    return (y, mask), st_
end

"""
    MLPAttn

ResNet item encoder plus a thin peer-attention residual, then a linear head.

Intended to sit after the usual embedding layer: `Chain(embed, MLPAttn(...))`.

# Forward signatures
- `(x, ps, st)` with `x` of shape `(features, batch)`: all observations are valid tokens.
- `((x, w), ps, st)`: `w` is a weight / boolean mask over the batch. Zero (or `false`)
  positions are treated as padded group-buffer slots and are ignored by attention via a
  rectangular key-padding mask of shape `(seq, 1, 1, 1)`, broadcast onto attention scores
  `(kv_len, q_len, nheads, 1)`. This is not a causal (triangular) mask.
"""
struct MLPAttn{N,B,H} <: LuxCore.AbstractLuxContainerLayer{(:encoder, :blocks, :head)}
    encoder::N
    blocks::B
    head::H
end

"""
    _key_padding_mask(w, seq)

Build a boolean key-padding mask for `MultiHeadAttention`.

Attention scores have shape `(kv_len, q_len, nheads, batch)`. A `true` entry keeps that
key position. Reshaping valid-token flags to `(seq, 1, 1, 1)` zeros out entire *columns*
of keys (padded buffer slots) for every query — a rectangular mask, not a causal triangle.
"""
_valid_tokens(v::AbstractVector{Bool}) = v
_valid_tokens(v::AbstractVector) = v .> zero(eltype(v))

function _key_padding_mask(w, seq::Int)
    return reshape(_valid_tokens(vec(w)), seq, 1, 1, 1)
end

function (m::MLPAttn)(x::AbstractArray, ps, st)
    z, st_n = m.encoder(x, ps.encoder, st.encoder)
    z, st_b = m.blocks(z, ps.blocks, st.blocks)
    y, st_h = m.head(z, ps.head, st.head)
    return y, (; encoder=st_n, blocks=st_b, head=st_h)
end
function (m::MLPAttn)((x, w)::Tuple, ps, st)
    z, st_n = m.encoder(x, ps.encoder, st.encoder)
    mask = _key_padding_mask(w, size(z, 2))
    z, st_b = m.blocks((z, mask), ps.blocks, st.blocks)
    y, st_h = m.head(z[1], ps.head, st.head)
    return y, (; encoder=st_n, blocks=st_b, head=st_h)
end

uses_batch_mask(::MLPAttn) = true

"""
    MLPAttnConfig(; kwargs...)

ResNet item encoder with a scaled multi-head attention residual over the batch / group.

The encoder is the same stem + residual blocks as ResNet (BatchNorm, not LayerNorm).
Attention is a single residual `x + scale * MHA(x)` — no extra LayerNorm tower and no
transformer FFN. A linear head produces the per-observation prediction.

When a padding mask is available (`w` from grouped loaders, or the infer `mask`),
the loss / eval / infer call sites pass `(x, w)` into the assembled `MaskedModel`.

# Arguments
- `act::Symbol`: Activation — `:relu`, `:gelu`, `:sigmoid`, or `:tanh` (default `:relu`).
- `hidden_size::Int`: Encoder / attention dimension (default `64`). Must be divisible by `nheads`.
- `stack_size::Int`: Number of residual blocks after the stem (default `1`), matching ResNet.
  `0` is a no-op (embedding width must equal `hidden_size`).
- `dropout::Float64`: Dropout in residual blocks and on the attention residual (default `0.0`).
- `nheads::Int`: Number of attention heads (default `4`).
- `n_attn_layers::Int`: Number of attention residuals (default `1`). `0` is encoder + head only.
- `attn_dropout::Float64`: Dropout on attention scores (default `0.0`).
- `attn_scale::Float32`: Initial attention residual scalar (default `0.1`).
- `MLE_tree_split::Bool`: Split output head for Gaussian MLE (default `false`).
"""
struct MLPAttnConfig <: Architecture
    act::Symbol
    hidden_size::Int
    stack_size::Int
    dropout::Float64
    nheads::Int
    n_attn_layers::Int
    attn_dropout::Float64
    attn_scale::Float32
    MLE_tree_split::Bool
end

function MLPAttnConfig(; kwargs...)
    args = Dict{Symbol,Any}(
        :act => :relu,
        :hidden_size => 64,
        :stack_size => 1,
        :dropout => 0.0,
        :nheads => 4,
        :n_attn_layers => 1,
        :attn_dropout => 0.0,
        :attn_scale => 0.1f0,
        :MLE_tree_split => false,
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    length(args_ignored) > 0 &&
        @warn "Following $(length(args_ignored)) provided arguments will be ignored: $(join(args_ignored, ", "))."

    args_default = setdiff(keys(args), keys(kwargs))
    length(args_default) > 0 &&
        @info "Following $(length(args_default)) arguments set to default: $(join(args_default, ", "))."

    for arg in intersect(keys(args), keys(kwargs))
        args[arg] = kwargs[arg]
    end

    return MLPAttnConfig(
        Symbol(args[:act]),
        args[:hidden_size],
        args[:stack_size],
        args[:dropout],
        args[:nheads],
        args[:n_attn_layers],
        args[:attn_dropout],
        Float32(args[:attn_scale]),
        args[:MLE_tree_split],
    )
end

"""
    _item_res_block(hsize, act, dropout)

Same residual block as ResNet: two Denses with BatchNorm and a skip.
"""
function _item_res_block(hsize::Int, act, dropout::Float64)
    layers = Any[Dense(hsize => hsize), BatchNorm(hsize, act)]
    dropout > 0 && push!(layers, Dropout(dropout))
    push!(layers, Dense(hsize => hsize))
    push!(layers, BatchNorm(hsize))
    return Chain(SkipConnection(Chain(layers...), +), BatchNorm(hsize, act))
end

"""
    _mlp_encoder(ins, hsize, act, stack_size, dropout)

Item-wise ResNet trunk without the prediction head.

- `stack_size == 0`: `NoOpLayer`. Requires `ins == hsize`.
- `stack_size >= 1`: `Dense(ins → hsize)` + BatchNorm+act, then `stack_size` residual blocks.
"""
function _mlp_encoder(ins::Int, hsize::Int, act, stack_size::Int, dropout::Float64)
    stack_size >= 0 || error("`stack_size` must be ≥ 0, got $stack_size.")
    if stack_size == 0
        ins == hsize || error(
            "`stack_size=0` passes the embedding through unchanged, so its width (`ins=$ins`) must equal `hidden_size` ($hsize).",
        )
        return NoOpLayer()
    end
    layers = Any[Dense(ins => hsize), BatchNorm(hsize, act)]
    for _ in 1:stack_size
        push!(layers, _item_res_block(hsize, act, dropout))
    end
    return Chain(layers...)
end

function _attn_blocks(
    hsize::Int, nheads::Int, n_attn_layers::Int, dropout::Float64, attn_dropout::Float64; attn_scale::Float32=0.1f0
)
    n_attn_layers >= 0 || error("`n_attn_layers` must be ≥ 0, got $n_attn_layers.")
    n_attn_layers == 0 && return NoOpLayer()
    blocks = [AttnResidual(hsize, nheads; dropout, attn_dropout, attn_scale) for _ in 1:n_attn_layers]
    return Chain(blocks...)
end

function _pred_head(hsize::Int, outsize::Int, MLE_tree_split::Bool)
    if MLE_tree_split
        iseven(outsize) || error("MLE_tree_split requires an even `outsize` (e.g., 2 for μ and σ). Got: $outsize")
        head_outsize = outsize ÷ 2
        return Parallel(vcat, Dense(hsize => head_outsize), Dense(hsize => head_outsize))
    end
    return Dense(hsize => outsize)
end

function _build_mlp_attn(ins::Int, outsize::Int, config::MLPAttnConfig)
    hsize = config.hidden_size
    nheads = config.nheads
    config.n_attn_layers >= 0 || error("`n_attn_layers` must be ≥ 0, got $(config.n_attn_layers).")
    if config.n_attn_layers > 0
        nheads > 0 || error("`nheads` must be ≥ 1 when `n_attn_layers` > 0, got $nheads.")
        hsize % nheads == 0 || error("`hidden_size` ($hsize) must be divisible by `nheads` ($nheads).")
    end

    act = get_activation(config.act)
    encoder = _mlp_encoder(ins, hsize, act, config.stack_size, config.dropout)
    blocks = _attn_blocks(
        hsize, nheads, config.n_attn_layers, config.dropout, config.attn_dropout; attn_scale=config.attn_scale
    )

    return MLPAttn(encoder, blocks, _pred_head(hsize, outsize, config.MLE_tree_split))
end

"""
    (config::MLPAttnConfig)(; ins, outsize)

Build an [`MLPAttn`](@ref) backbone from `config`. `fit` prepends embeddings via
`MaskedModel(embed, core)` when a padding mask must reach attention;
otherwise `Chain(embed, core)` as for the other architectures.
"""
function (config::MLPAttnConfig)(; ins, outsize, kwargs...)
    return _build_mlp_attn(ins, outsize, config)
end

end
