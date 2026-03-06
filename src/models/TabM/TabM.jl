module TabM

export TabMConfig

using Random
using Lux
using LuxCore
using NNlib

import ..Losses: get_loss_type, GaussianMLE
import ..Models: Architecture
import ..Embeddings: PeriodicEmbeddings, LinearEmbeddings, PiecewiseLinearEmbeddings

include("layers.jl")

"""
    FlattenEmb()

Reshapes 3D embedding output `(d_embedding, n_features, batch)` → `(d_embedding * n_features, batch)`.
Passes through 2D input unchanged.
"""
struct FlattenEmb <: Lux.AbstractLuxLayer end
(::FlattenEmb)(x::AbstractArray{T,3}, ps, st) where {T} = reshape(x, :, size(x, 3)), st
(::FlattenEmb)(x::AbstractMatrix, ps, st) = x, st

function _batch_ensemble_backbone(;
        d_in::Int, n_blocks::Int, d_block::Int, dropout::Float64,
        k::Int, scaling_init::Symbol, d_features::Vector{Int})
    layers = Any[]
    for i in 1:n_blocks
        d_in_i = (i == 1) ? d_in : d_block
        if i == 1
            push!(layers, LinearBatchEnsemble(d_in_i, d_block;
                k, scaling_init = (scaling_init, :ones),
                first_scaling_init_chunks = d_features))
        else
            push!(layers, LinearBatchEnsemble(d_in_i, d_block;
                k, scaling_init = :ones))
        end
        push!(layers, WrappedFunction(_broadcast_relu))
        dropout > 0 && push!(layers, Dropout(dropout))
    end
    return layers
end

function _mini_ensemble_backbone(;
        d_in::Int, n_blocks::Int, d_block::Int, dropout::Float64,
        k::Int, scaling_init::Symbol, d_features::Vector{Int})
    layers = Any[ScaleEnsemble(k, d_in;
        init = scaling_init, init_chunks = d_features, bias = false)]
    for i in 1:n_blocks
        d_in_i = (i == 1) ? d_in : d_block
        push!(layers, SharedDense(d_in_i, d_block))
        push!(layers, WrappedFunction(_broadcast_relu))
        dropout > 0 && push!(layers, Dropout(dropout))
    end
    return layers
end

function _packed_ensemble_backbone(;
        d_in::Int, n_blocks::Int, d_block::Int, dropout::Float64, k::Int)
    layers = Any[]
    for i in 1:n_blocks
        d_in_i = (i == 1) ? d_in : d_block
        push!(layers, LinearEnsemble(d_in_i, d_block, k))
        push!(layers, WrappedFunction(_broadcast_relu))
        dropout > 0 && push!(layers, Dropout(dropout))
    end
    return layers
end

"""
    TabMConfig(; kwargs...)

Configuration for TabM ensemble architectures (Gorishniy et al., ICLR 2025).

This implements the TabM♠ variant (shared training batches), where all k ensemble
members see the same training batch.

The chain output is 3D: `(outsize, k, batch)`. Ensemble averaging is handled by
`Losses.reduce_pred` during training and `_reduce` in `infer.jl` at inference.

# Arguments
- `k::Int`: Number of ensemble members (default `32`).
- `n_blocks::Int`: Number of MLP blocks (default `2` with embeddings, `3` without).
- `d_block::Int`: Hidden dimension per block (default `512`).
- `dropout::Float64`: Dropout rate (default `0.1`).
- `arch_type::Symbol`: `:tabm`, `:tabm_mini`, or `:tabm_packed` (default `:tabm`).
- `scaling_init::Symbol`: Init for ensemble scaling vectors — `:random_signs`, `:normal`, or `:ones` (default `:normal` with embeddings, `:random_signs` without).
- `MLE_tree_split::Bool`: Split output head for Gaussian MLE (default `false`).
- `use_embeddings::Bool`: Apply feature embeddings before the backbone (default `false`).
- `d_embedding::Int`: Embedding dimension per feature (default `24`).
- `embedding_type::Symbol`: `:periodic`, `:linear`, or `:piecewise` (default `:periodic`).
- `bins::Union{Nothing,Vector{Vector{Float32}}}`: Bin edges for piecewise embeddings (default `nothing`).

# Callable
    config(; nfeats, outsize) → Lux.Chain

- `nfeats::Int`: Number of input features.
- `outsize::Int`: Output dimension.
"""
struct TabMConfig <: Architecture
    k::Int
    n_blocks::Int
    d_block::Int
    dropout::Float64
    arch_type::Symbol
    scaling_init::Symbol
    MLE_tree_split::Bool
    use_embeddings::Bool
    d_embedding::Int
    embedding_type::Symbol
    bins::Union{Nothing,Vector{Vector{Float32}}}
end

function TabMConfig(; kwargs...)
    has_embeddings = get(kwargs, :use_embeddings, false)

    args = Dict{Symbol,Any}(
        :k => 32,
        :n_blocks => has_embeddings ? 2 : 3,
        :d_block => 512,
        :dropout => 0.1,
        :arch_type => :tabm,
        :scaling_init => has_embeddings ? :normal : :random_signs,
        :MLE_tree_split => false,
        :use_embeddings => false,
        :d_embedding => 24,
        :embedding_type => :periodic,
        :bins => nothing,
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

    return TabMConfig(
        args[:k], args[:n_blocks], args[:d_block], args[:dropout],
        Symbol(args[:arch_type]), Symbol(args[:scaling_init]),
        args[:MLE_tree_split], args[:use_embeddings],
        args[:d_embedding], Symbol(args[:embedding_type]), args[:bins],
    )
end

function (config::TabMConfig)(; nfeats, outsize)
    @assert config.k > 0 "k must be > 0, got $(config.k)"
    @assert nfeats > 0 "nfeats must be > 0, got $nfeats"
    @assert outsize > 0 "outsize must be > 0, got $outsize"

    k = config.k
    d_block = config.d_block

    if config.use_embeddings
        emb_layer = if config.embedding_type == :periodic
            PeriodicEmbeddings(nfeats, config.d_embedding)
        elseif config.embedding_type == :linear
            LinearEmbeddings(nfeats, config.d_embedding)
        elseif config.embedding_type == :piecewise
            @assert config.bins !== nothing "Piecewise embeddings require bins"
            @assert length(config.bins) == nfeats "Expected $nfeats bin vectors, got $(length(config.bins))"
            PiecewiseLinearEmbeddings(config.bins, config.d_embedding)
        else
            error("Unsupported embedding type: $(config.embedding_type)")
        end
        feature_layers = [emb_layer, FlattenEmb()]
        d_in = nfeats * config.d_embedding
        d_features = fill(config.d_embedding, nfeats)
        effective_scaling_init = :normal
    else
        feature_layers = []
        d_in = nfeats
        d_features = ones(Int, nfeats)
        effective_scaling_init = config.scaling_init
    end

    bb = if config.arch_type == :tabm
        _batch_ensemble_backbone(; d_in, n_blocks=config.n_blocks,
            d_block, dropout=config.dropout, k,
            scaling_init=effective_scaling_init, d_features)
    elseif config.arch_type == :tabm_mini
        _mini_ensemble_backbone(; d_in, n_blocks=config.n_blocks,
            d_block, dropout=config.dropout, k,
            scaling_init=effective_scaling_init, d_features)
    elseif config.arch_type == :tabm_packed
        _packed_ensemble_backbone(; d_in, n_blocks=config.n_blocks,
            d_block, dropout=config.dropout, k)
    else
        error("Unknown arch_type: $(config.arch_type)")
    end

    head = if config.MLE_tree_split && outsize == 2
        split_out = outsize ÷ 2
        Parallel(vcat,
            LinearEnsemble(d_block, split_out, k),
            LinearEnsemble(d_block, split_out, k))
    else
        LinearEnsemble(d_block, outsize, k)
    end

    return Chain(feature_layers..., EnsembleView(k), bb..., head)
end

end