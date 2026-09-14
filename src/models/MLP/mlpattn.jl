"""
    MLPAttn

ResNet-style encoder (BatchNorm residual MLP) plus a thin peer-attention residual,
then a linear head.

Scale: BatchNorm on the item stream (same recipe as [`ResNetConfig`](@ref NeuroTabModels.Models.ResNet.ResNetConfig)).
`n_attn_layers=0` is that encoder plus a Glorot head — the ResNet ablation.

Intended to sit after the usual embedding layer: `Chain(embed, MLPAttn(...))`.

# Forward signatures
- `(x, ps, st)` with `x` of shape `(features, batch)`: all observations are valid tokens.
- `((x, w), ps, st)`: `w` is a weight / boolean mask over the batch. Zero (or `false`)
  positions are treated as padded group-buffer slots and are ignored by attention via a
  rectangular key-padding mask of shape `(seq, 1, 1, 1)`, broadcast onto attention scores
  `(kv_len, q_len, nheads, 1)`. This is not a causal (triangular) mask.
  Encoder BatchNorm uses the same valid-token flags so pads do not enter mean/var
  (or running stats). The attention mask still zeros padded *keys*.
"""
struct MLPAttn{N,B,H} <: LuxCore.AbstractLuxContainerLayer{(:encoder, :blocks, :head)}
    encoder::N
    blocks::B
    head::H
end

function (m::MLPAttn)(x::AbstractArray, ps, st)
    z, st_n = m.encoder(x, ps.encoder, st.encoder)
    z, st_b = m.blocks(z, ps.blocks, st.blocks)
    y, st_h = m.head(z, ps.head, st.head)
    return y, (; encoder=st_n, blocks=st_b, head=st_h)
end
function (m::MLPAttn)((x, w)::Tuple, ps, st)
    valid = _valid_tokens(vec(w))
    z, st_n = m.encoder((x, valid), ps.encoder, st.encoder)
    z = _untuple(z)
    mask = reshape(valid, size(z, 2), 1, 1, 1)
    z, st_b = m.blocks((z, mask), ps.blocks, st.blocks)
    y, st_h = m.head(z[1], ps.head, st.head)
    return y, (; encoder=st_n, blocks=st_b, head=st_h)
end

uses_batch_mask(::MLPAttn) = true

"""
    MLPAttnConfig(; kwargs...)

Item-wise residual encoder with peer attention over the batch / group.

The encoder is a BatchNorm residual MLP like [`ResNetConfig`](@ref NeuroTabModels.Models.ResNet.ResNetConfig), but each BN
restricts mean/var to valid tokens when a padding mask is passed (grouped loaders).
Attention is shared Q=K via `NNlib.dot_product_attention`, values = encoder tokens,
residual-added with a learned scalar (`attn_scale`, default `0.1`) so peer mixing
starts small. The head is Glorot `Dense` like ResNet. `n_attn_layers=0` should track
ResNet on ungrouped data.

When a padding mask is available (`w` from grouped loaders, or the infer `mask`),
the loss / eval / infer call sites pass `(x, w)` into the assembled `MaskedModel`.

# Arguments
- `act::Symbol`: Activation — `:relu`, `:gelu`, `:sigmoid`, or `:tanh` (default `:relu`).
- `hidden_size::Int`: Encoder / attention dimension (default `64`). Must be divisible by `nheads`.
- `stack_size::Int`: Number of residual blocks after the stem (default `1`).
  `0` is a no-op (embedding width must equal `hidden_size`).
- `dropout::Float64`: Dropout in residual blocks and on the attention residual (default `0.0`).
- `nheads::Int`: Number of attention heads (default `4`).
- `n_attn_layers::Int`: Number of attention residuals (default `1`). `0` is encoder + head only.
- `attn_dropout::Float64`: Dropout on attention scores (default `0.0`).
- `attn_scale::Float32`: Initial value of the learned attention residual scale
  (default `0.1`). Mixing is `x + scale * Attn`.
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
    )
end

"""
    _res_block(hsize, act, dropout)

ResNet residual block with masked BatchNorm so grouped pads can be ignored.
"""
function _res_block(hsize::Int, act, dropout::Float64)
    layers = Any[CarryMask(Dense(hsize => hsize)), MaskedBatchNorm(hsize, act)]
    dropout > 0 && push!(layers, CarryMask(Dropout(dropout)))
    push!(layers, CarryMask(Dense(hsize => hsize)))
    push!(layers, MaskedBatchNorm(hsize))
    return Chain(MaskSkip(Chain(layers...)), MaskedBatchNorm(hsize, act))
end

"""
    _mlp_encoder(ins, hsize, act, stack_size, dropout)

ResNet trunk without the prediction `Dense`. Accepts `x` or `(x, valid)` so
BatchNorm can skip padded group-buffer slots.

- `stack_size == 0`: `NoOpLayer`. Requires `ins == hsize`.
- `stack_size >= 1`: `Dense(ins → hsize)` + MaskedBN+act, then `stack_size` residual blocks.
"""
function _mlp_encoder(ins::Int, hsize::Int, act, stack_size::Int, dropout::Float64)
    stack_size >= 0 || error("`stack_size` must be ≥ 0, got $stack_size.")
    if stack_size == 0
        ins == hsize || error(
            "`stack_size=0` passes the embedding through unchanged, so its width (`ins=$ins`) must equal `hidden_size` ($hsize).",
        )
        return NoOpLayer()
    end
    layers = Any[CarryMask(Dense(ins => hsize)), MaskedBatchNorm(hsize, act)]
    for _ in 1:stack_size
        push!(layers, _res_block(hsize, act, dropout))
    end
    return Chain(layers...)
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
    blocks = _attn_blocks(hsize, nheads, config.n_attn_layers, config.dropout, config.attn_dropout, config.attn_scale)

    return MLPAttn(encoder, blocks, _pred_head(hsize, outsize))
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
