"""
    NeuroTreeAttn

NeuroTree encoder (per-observation numerical embeddings) followed by a
per-channel peer-attention residual, then a prediction head.

Intended to sit after the usual embedding layer: `Chain(embed, NeuroTreeAttn(...))`.
Same role as `MLPAttn`: a per-row map into `hidden_size`, peer attention over the
batch / group. For a scalar head the `k` ensembles are kept through the loss,
same layout as `NeuroTreeConfig`.

The encoder width is NeuroTree's `k` axis, not `outs`:
`NeuroTree(ins => 1; k = hidden_size)`. Each hidden channel is its own
ensemble (independent splits and leaf values). Putting `hidden_size` on `outs`
with `k = 1` would share one routing among all channels. Native layout is
`(1, k, batch)`; `FlattenLayer` only drops the singleton `outs` axis so attention sees
`(hidden_size, batch)`. After attention, the `k` axis is restored to
`(1, k, batch)` so the loss trains each channel as an independent predictor,
same as `NeuroTreeConfig`. A Dense+BatchNorm head would collapse that axis.

Those tokens are the sequence of a single batch / group, identical to MLPAttn.

# Forward signatures
- `(x, ps, st)` with `x` of shape `(features, batch)`: all observations are valid tokens.
- `((x, w), ps, st)`: `w` is a weight / boolean mask over the batch. Zero (or `false`)
  positions are treated as padded group-buffer slots and are ignored by
  attention via a rectangular key-padding mask.
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
    valid = _valid_tokens(vec(w))
    z, st_n = m.encoder((x, valid), ps.encoder, st.encoder)
    z = _untuple(z)
    mask = reshape(valid, size(z, 2), 1, 1, 1)
    z, st_b = m.blocks((z, mask), ps.blocks, st.blocks)
    y, st_h = m.head(z[1], ps.head, st.head)
    return y, (; encoder=st_n, blocks=st_b, head=st_h)
end

uses_batch_mask(::NeuroTreeAttn) = true

"""
    NeuroTreeAttnConfig(; kwargs...)

Configuration for a NeuroTree encoder plus batch-level transformer attention.

The tree stem is `NeuroTree(ins => 1; k = hidden_size)`: `hidden_size`
independent ensembles (own splits per channel). `FlattenLayer` reshapes
`(1, k, batch) → (k, batch)` for attention, then the head restores
`(1, k, batch)` so MSE trains each ensemble against `y` (inference means
over `k`). Peer attention is **per ensemble channel** (one head per `k`, no
Dense QK mixing), residual-added with `attn_scale` (default `0`). There is no
transformer FFN.

When a padding mask is available (`w` from grouped loaders, or the infer `mask`),
the loss / eval / infer call sites pass `(x, w)` into the assembled `MaskedModel`.

# Arguments
- `tree_type::Symbol`: `:binary` or `:oblivious` (default `:binary`).
- `actA::Symbol`: Feature activation on split weights. One of `:identity`, `:tanh`,
  `:hardtanh`, or `:tanhshrink` (default `:identity`).
- `depth::Int`: Tree depth (default `4`). Controls the number of leaves (`2^depth`),
  which is an internal routing axis — not the hidden width.
- `ntrees::Int`: Number of trees averaged in each of the `k` hidden ensembles (default `32`).
- `hidden_size::Int`: Encoding / attention dimension (default `64`). Equals encoder
  NeuroTree `k` (`outs = 1`), so each channel has its own splits. Peer attention
  uses one head per channel (`nheads` is ignored).
- `stack_size::Int`: Encoder depth (default `1`). `0` is a no-op (embedding width must
  equal `hidden_size`). `1` is a single `NeuroTree` + flatten. Each extra
  layer is a residual `NeuroTree` of width `hidden_size`, with optional dropout.
- `scaler::Bool`: Apply softplus scaling on tree logits (default `true`).
- `init_scale::Float32`: Leaf weight init scale (default `0.1`).
- `dropout::Float64`: Dropout after extra encoder layers only (`stack_size ≥ 2`).
  Not applied to the attention residual (those tokens are the `k` predictions).
- `nheads::Int`: Ignored. Peer attention uses one head per ensemble channel.
- `n_attn_layers::Int`: Number of attention residuals (default `1`). `0` skips attention.
- `attn_dropout::Float64`: Dropout on attention scores (default `0.0`).
- `attn_scale::Float32`: Initial value of the learned attention residual scale
  (default `0.0`). Mixing is `x + scale * Attn` with **one head per ensemble
  channel** (config `nheads` is not used). Default `0` so `n_attn_layers=1`
  starts as `n_attn_layers=0`. Raise it if grouped peers should mix.
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
    dropout::Float64
    nheads::Int
    n_attn_layers::Int
    attn_dropout::Float64
    attn_scale::Float32
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
        :dropout => 0.0,
        :nheads => 4,
        :n_attn_layers => 1,
        :attn_dropout => 0.0,
        :attn_scale => 0.0f0,
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
        args[:dropout],
        args[:nheads],
        args[:n_attn_layers],
        args[:attn_dropout],
        Float32(args[:attn_scale]),
    )
end

function _attn_tree_kwargs(config::NeuroTreeAttnConfig)
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
    _tree_attn_block(ins, hsize, tree_kwargs)

One NeuroTree encoder block: `k = hsize` independent ensembles, `outs = 1`
(scalar per leaf, own splits per channel). `FlattenLayer` only drops the
singleton `outs` axis (`(1, k, batch) → (k, batch)`) so attention sees a 2D
token matrix. Wrapped in `CarryMask` so a padding flag still reaches attention
after the encoder.
"""
function _tree_attn_block(ins::Int, hsize::Int, tree_kwargs)
    return CarryMask(Chain(NeuroTree(ins => 1; k=hsize, tree_kwargs...), FlattenLayer()))
end

# Restore NeuroTree layout `(1, k, batch)` so the loss trains each ensemble.
_k_as_ensemble(x::AbstractMatrix) = reshape(x, 1, size(x, 1), size(x, 2))

function _tree_attn_head(hsize::Int, outsize::Int)
    outsize == 1 && return WrappedFunction(_k_as_ensemble)
    return _pred_head(hsize, outsize)
end

"""
    _tree_attn_encoder(ins, hsize, stack_size, dropout, tree_kwargs)

Per-observation map into the attention width `hsize`.

Each `NeuroTree` produces `(1, k, batch)` with `k = hsize`. `FlattenLayer`
reshapes to `(hsize, batch)`. Same adapter as stacked NeuroTree hidden layers.

- `stack_size == 0`: `NoOpLayer`. Requires `ins == hsize` so the NeuroTab embedding
  block can be the sole numerical embedding.
- `stack_size == 1`: `_tree_attn_block(ins, hsize)`. No encoder dropout.
- `stack_size >= 2`: that stem, then `stack_size - 1` residual `NeuroTree` blocks
  of width `hsize`, with optional dropout after each residual.
"""
function _tree_attn_encoder(ins::Int, hsize::Int, stack_size::Int, dropout::Float64, tree_kwargs)
    stack_size >= 0 || error("`stack_size` must be ≥ 0, got $stack_size.")
    if stack_size == 0
        ins == hsize || error(
            "`stack_size=0` passes the embedding through unchanged, so its width (`ins=$ins`) must equal `hidden_size` ($hsize).",
        )
        return NoOpLayer()
    end
    layers = Any[_tree_attn_block(ins, hsize, tree_kwargs)]
    for _ in 2:stack_size
        push!(layers, MaskSkip(_tree_attn_block(hsize, hsize, tree_kwargs)))
        dropout > 0 && push!(layers, CarryMask(Dropout(dropout)))
    end
    return Chain(layers...)
end

function _build_neurotree_attn(ins::Int, outsize::Int, config::NeuroTreeAttnConfig)
    hsize = config.hidden_size
    config.n_attn_layers >= 0 || error("`n_attn_layers` must be ≥ 0, got $(config.n_attn_layers).")

    encoder = _tree_attn_encoder(ins, hsize, config.stack_size, config.dropout, _attn_tree_kwargs(config))
    blocks = _tree_attn_blocks(hsize, config.n_attn_layers, config.attn_dropout, config.attn_scale)

    return NeuroTreeAttn(encoder, blocks, _tree_attn_head(hsize, outsize))
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
