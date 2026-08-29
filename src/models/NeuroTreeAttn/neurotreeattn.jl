module NeuroTreeAttn

export NeuroTreeAttnConfig

using Lux
using LuxCore

import ..Models: Architecture, get_activation, uses_batch_mask
import ..NeuroTrees: NeuroTree, act_dict
import ..MLPAttn

"""
    NeuroTreeAttn

NeuroTree encoder (per-observation numerical embeddings) followed by transformer blocks
that mix information across observations in the batch, then a linear prediction head.

Intended to sit after the usual embedding layer: `Chain(embed, NeuroTreeAttn(...))`.

Each observation is mapped to a `hidden_size` token by a differentiable tree ensemble
(`NeuroTree` with `k = hidden_size`, then `FlattenLayer`). Those tokens are the sequence
of a single batch / group, identical to MLPAttn.

# Forward signatures
- `(x, ps, st)` with `x` of shape `(features, batch)`: all observations are valid tokens.
- `((x, w), ps, st)`: `w` is a weight / boolean mask over the batch. Zero (or `false`)
  positions are treated as padded group-buffer slots and are ignored by attention via a
  rectangular key-padding mask.
"""
struct NeuroTreeAttn{N,B,H} <: LuxCore.AbstractLuxContainerLayer{(:encoder, :blocks, :head)}
    encoder::N
    blocks::B
    head::H
end

function (m::NeuroTreeAttn)(x::AbstractArray, ps, st)
    z, st_n = m.encoder(x, ps.encoder, st.encoder)
    z, st_b = m.blocks(z, ps.blocks, st.blocks)
    y, st_h = m.head(z, ps.head, st.head)
    return y, (; encoder=st_n, blocks=st_b, head=st_h)
end
function (m::NeuroTreeAttn)((x, w)::Tuple, ps, st)
    z, st_n = m.encoder(x, ps.encoder, st.encoder)
    mask = MLPAttn._key_padding_mask(w, size(z, 2))
    z, st_b = m.blocks((z, mask), ps.blocks, st.blocks)
    y, st_h = m.head(z[1], ps.head, st.head)
    return y, (; encoder=st_n, blocks=st_b, head=st_h)
end

uses_batch_mask(::NeuroTreeAttn) = true

"""
    NeuroTreeAttnConfig(; kwargs...)

Configuration for a NeuroTree encoder plus batch-level transformer attention.

The tree root maps each observation to a `hidden_size` embedding (`k = hidden_size`
scalar trees, flattened). Those embeddings are the tokens of a single sequence (the
current batch / group). Transformer blocks mix signal across observations, then a
linear head produces the per-observation prediction.

When a padding mask is available (`w` from grouped loaders, or the infer `mask`),
the loss / eval / infer call sites pass `(x, w)` into the assembled `MaskedModel`.

# Arguments
- `tree_type::Symbol`: `:binary` or `:oblivious` (default `:binary`).
- `actA::Symbol`: Feature activation on split weights. One of `:identity`, `:tanh`,
  `:hardtanh`, or `:tanhshrink` (default `:identity`).
- `depth::Int`: Tree depth (default `4`).
- `ntrees::Int`: Number of trees per encoder layer (default `32`).
- `hidden_size::Int`: Encoding / attention dimension (default `64`). Must be divisible
  by `nheads`. Each observation is encoded by `hidden_size` tree ensembles (`k`).
- `stack_size::Int`: Encoder depth (default `1`). `0` is a no-op (embedding width must
  equal `hidden_size`). `1` is a single `NeuroTree` + flatten. Each extra layer is a
  residual `NeuroTree` of width `hidden_size`, with optional dropout.
- `scaler::Bool`: Apply softplus scaling on tree logits (default `true`).
- `init_scale::Float32`: Leaf weight init scale (default `0.1`).
- `act::Symbol`: Transformer FFN activation — `:relu`, `:gelu`, `:sigmoid`, or `:tanh`
  (default `:relu`).
- `dropout::Float64`: Dropout after extra encoder layers and after attention/FFN
  residuals (default `0.0`).
- `nheads::Int`: Number of attention heads (default `4`).
- `n_attn_layers::Int`: Number of transformer blocks (default `1`).
- `attn_dropout::Float64`: Dropout on attention scores (default `0.0`).
- `ffn_hidden::Int`: Transformer FFN inner width (default `0` → `hidden_size`).
"""
struct NeuroTreeAttnConfig <: Architecture
    tree_type::Symbol
    actA::Symbol
    depth::Int
    ntrees::Int
    hidden_size::Int
    stack_size::Int
    scaler::Bool
    init_scale::Float32
    act::Symbol
    dropout::Float64
    nheads::Int
    n_attn_layers::Int
    attn_dropout::Float64
    ffn_hidden::Int
end

function NeuroTreeAttnConfig(; kwargs...)
    args = Dict{Symbol,Any}(
        :tree_type => :binary,
        :actA => :identity,
        :depth => 4,
        :ntrees => 32,
        :hidden_size => 64,
        :stack_size => 1,
        :scaler => true,
        :init_scale => 0.1,
        :act => :relu,
        :dropout => 0.0,
        :nheads => 4,
        :n_attn_layers => 1,
        :attn_dropout => 0.0,
        :ffn_hidden => 0,
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

    return NeuroTreeAttnConfig(
        Symbol(args[:tree_type]),
        Symbol(args[:actA]),
        args[:depth],
        args[:ntrees],
        args[:hidden_size],
        args[:stack_size],
        args[:scaler],
        args[:init_scale],
        Symbol(args[:act]),
        args[:dropout],
        args[:nheads],
        args[:n_attn_layers],
        args[:attn_dropout],
        args[:ffn_hidden],
    )
end

function _tree_kwargs(config::NeuroTreeAttnConfig)
    return (;
        tree_type=config.tree_type,
        depth=config.depth,
        trees=config.ntrees,
        actA=act_dict[config.actA],
        scaler=config.scaler,
        init_scale=config.init_scale,
    )
end

"""
    _tree_encoder(ins, hsize, stack_size, dropout, tree_kwargs)

Per-observation map into the attention width `hsize`.

Each `NeuroTree` produces shape `(1, k, batch)` with `k = hsize`; `FlattenLayer`
turns that into `(hsize, batch)` tokens for attention.

- `stack_size == 0`: `NoOpLayer`. Requires `ins == hsize` so the NeuroTab embedding
  block can be the sole numerical embedding.
- `stack_size == 1`: `NeuroTree(ins → 1; k=hsize)` + flatten. No encoder dropout:
  each `TransformerBlock` already starts with pre-norm LayerNorm.
- `stack_size >= 2`: that stem, then `stack_size - 1` residual `NeuroTree` blocks
  of width `hsize`, with optional dropout after each residual.
"""
function _tree_encoder(ins::Int, hsize::Int, stack_size::Int, dropout::Float64, tree_kwargs)
    stack_size >= 0 || error("`stack_size` must be ≥ 0, got $stack_size.")
    if stack_size == 0
        ins == hsize || error(
            "`stack_size=0` passes the embedding through unchanged, so its width (`ins=$ins`) must equal `hidden_size` ($hsize).",
        )
        return NoOpLayer()
    end
    layers = Any[NeuroTree(ins => 1; k=hsize, tree_kwargs...), FlattenLayer()]
    for _ in 2:stack_size
        push!(layers, SkipConnection(Chain(NeuroTree(hsize => 1; k=hsize, tree_kwargs...), FlattenLayer()), +))
        dropout > 0 && push!(layers, Dropout(dropout))
    end
    return Chain(layers...)
end

function _build_neurotree_attn(ins::Int, outsize::Int, config::NeuroTreeAttnConfig)
    hsize = config.hidden_size
    nheads = config.nheads
    hsize % nheads == 0 || error("`hidden_size` ($hsize) must be divisible by `nheads` ($nheads).")
    config.n_attn_layers >= 1 || error("`n_attn_layers` must be ≥ 1, got $(config.n_attn_layers).")

    act = get_activation(config.act)
    ffn_hidden = config.ffn_hidden > 0 ? config.ffn_hidden : hsize
    encoder = _tree_encoder(ins, hsize, config.stack_size, config.dropout, _tree_kwargs(config))
    blocks = MLPAttn._attn_blocks(
        hsize, nheads, act, config.n_attn_layers, config.dropout, config.attn_dropout, ffn_hidden
    )

    head = Dense(hsize => outsize)

    return NeuroTreeAttn(encoder, blocks, head)
end

"""
    (config::NeuroTreeAttnConfig)(; ins, outsize)

Build a [`NeuroTreeAttn`](@ref) backbone from `config`. `fit` prepends embeddings via
`MaskedModel(embed, core)` when a padding mask must reach attention;
otherwise `Chain(embed, core)` as for the other architectures.
"""
function (config::NeuroTreeAttnConfig)(; ins, outsize, kwargs...)
    return _build_neurotree_attn(ins, outsize, config)
end

end
