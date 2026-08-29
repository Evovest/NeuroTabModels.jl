module MLPAttn

export MLPAttnConfig

using Lux
using LuxCore

import ..Models: Architecture, get_activation, uses_batch_mask

"""
    TransformerBlock(hsize, nheads, act; dropout=0.0, attn_dropout=0.0, ffn_hidden=hsize)

Pre-norm transformer block: LayerNorm → multi-head self-attention → residual,
then LayerNorm → FFN → residual.

Inputs are `(hidden, seq)` feature-first matrices. An optional key-padding mask
may be passed as `(x, mask)` so padded group-buffer slots are ignored.
"""
struct TransformerBlock{N1,A,D1,N2,F,D2} <: LuxCore.AbstractLuxContainerLayer{(
    :norm1, :mha, :drop1, :norm2, :ffn, :drop2
)}
    norm1::N1
    mha::A
    drop1::D1
    norm2::N2
    ffn::F
    drop2::D2
end

function TransformerBlock(
    hsize::Int, nheads::Int, act; dropout::Float64=0.0, attn_dropout::Float64=0.0, ffn_hidden::Int=hsize
)
    return TransformerBlock(
        LayerNorm((hsize,); dims=1),
        MultiHeadAttention(hsize; nheads, attention_dropout_probability=Float32(attn_dropout)),
        Dropout(dropout),
        LayerNorm((hsize,); dims=1),
        Chain(Dense(hsize => ffn_hidden, act), Dense(ffn_hidden => hsize)),
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

function _block(l::TransformerBlock, x::AbstractMatrix, mask, ps, st)
    h, st_n1 = l.norm1(x, ps.norm1, st.norm1)
    a, st_a = _mha(l.mha, h, mask, ps.mha, st.mha)
    a, st_d1 = l.drop1(a, ps.drop1, st.drop1)
    x = x .+ a
    h, st_n2 = l.norm2(x, ps.norm2, st.norm2)
    f, st_f = l.ffn(h, ps.ffn, st.ffn)
    f, st_d2 = l.drop2(f, ps.drop2, st.drop2)
    y = x .+ f
    return y, (; norm1=st_n1, mha=st_a, drop1=st_d1, norm2=st_n2, ffn=st_f, drop2=st_d2)
end

(l::TransformerBlock)(x::AbstractMatrix, ps, st) = _block(l, x, nothing, ps, st)
function (l::TransformerBlock)((x, mask)::Tuple, ps, st)
    y, st_ = _block(l, x, mask, ps, st)
    return (y, mask), st_
end

"""
    MLPAttn

MLP encoder (per-observation numerical embeddings) followed by transformer blocks
that mix information across observations in the batch, then a linear prediction head.

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

Configuration for an MLP encoder plus batch-level transformer attention.

The MLP root maps each observation to a `hidden_size` embedding. Those embeddings are the
tokens of a single sequence (the current batch / group). Transformer blocks mix signal
across observations, then a linear head produces the per-observation prediction.

When a padding mask is available (`w` from grouped loaders, or the infer `mask`),
the loss / eval / infer call sites pass `(x, w)` into the assembled `MaskedModel`.

# Arguments
- `act::Symbol`: Activation — `:relu`, `:gelu`, `:sigmoid`, or `:tanh` (default `:relu`).
- `hidden_size::Int`: Embedding / attention dimension (default `64`). Must be divisible by `nheads`.
- `stack_size::Int`: Encoder depth (default `1`). `0` is a no-op (embedding width must
  equal `hidden_size`). `1` is a linear map `ins → hidden_size`. Each extra layer is
  pre-norm `LayerNorm` + `Dense(hidden_size → hidden_size)` with optional dropout.
- `dropout::Float64`: Dropout in the encoder and after attention/FFN residuals (default `0.0`).
- `nheads::Int`: Number of attention heads (default `4`).
- `n_attn_layers::Int`: Number of transformer blocks (default `1`).
- `attn_dropout::Float64`: Dropout on attention scores (default `0.0`).
- `ffn_hidden::Int`: Transformer FFN inner width (default `0` → `hidden_size`).
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
    ffn_hidden::Int
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
        :ffn_hidden => 0,
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
        args[:ffn_hidden],
        args[:MLE_tree_split],
    )
end

"""
    _mlp_encoder(ins, hsize, act, stack_size, dropout)

Per-observation map into the attention width `hsize`.

- `stack_size == 0`: `NoOpLayer`. Requires `ins == hsize` so the NeuroTab embedding
  block can be the sole numerical embedding.
- `stack_size == 1`: `Dense(ins → hsize)` only. No encoder LayerNorm: each
  `TransformerBlock` already starts with pre-norm LayerNorm.
- `stack_size >= 2`: that projection, then `stack_size - 1` blocks of
  `LayerNorm(act) → Dense → Dropout`. Dropout sits after the extra Denses, not on
  the stem projection (the transformer residual path already regularizes).
"""
function _mlp_encoder(ins::Int, hsize::Int, act, stack_size::Int, dropout::Float64)
    stack_size >= 0 || error("`stack_size` must be ≥ 0, got $stack_size.")
    if stack_size == 0
        ins == hsize || error(
            "`stack_size=0` passes the embedding through unchanged, so its width (`ins=$ins`) must equal `hidden_size` ($hsize).",
        )
        return NoOpLayer()
    end
    layers = Any[Dense(ins => hsize)]
    for _ in 2:stack_size
        push!(layers, LayerNorm((hsize,), act; dims=1))
        push!(layers, Dense(hsize => hsize))
        dropout > 0 && push!(layers, Dropout(dropout))
    end
    return Chain(layers...)
end

function _attn_blocks(
    hsize::Int, nheads::Int, act, n_attn_layers::Int, dropout::Float64, attn_dropout::Float64, ffn_hidden::Int
)
    blocks = [TransformerBlock(hsize, nheads, act; dropout, attn_dropout, ffn_hidden) for _ in 1:n_attn_layers]
    return Chain(blocks...)
end

function _build_mlp_attn(ins::Int, outsize::Int, config::MLPAttnConfig)
    hsize = config.hidden_size
    nheads = config.nheads
    hsize % nheads == 0 || error("`hidden_size` ($hsize) must be divisible by `nheads` ($nheads).")
    config.n_attn_layers >= 1 || error("`n_attn_layers` must be ≥ 1, got $(config.n_attn_layers).")

    act = get_activation(config.act)
    ffn_hidden = config.ffn_hidden > 0 ? config.ffn_hidden : hsize
    encoder = _mlp_encoder(ins, hsize, act, config.stack_size, config.dropout)
    blocks = _attn_blocks(hsize, nheads, act, config.n_attn_layers, config.dropout, config.attn_dropout, ffn_hidden)

    head = if config.MLE_tree_split
        iseven(outsize) || error("MLE_tree_split requires an even `outsize` (e.g., 2 for μ and σ). Got: $outsize")
        head_outsize = outsize ÷ 2
        Parallel(vcat, Dense(hsize => head_outsize), Dense(hsize => head_outsize))
    else
        Dense(hsize => outsize)
    end

    return MLPAttn(encoder, blocks, head)
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
