module TabM

export TabMConfig

using Random
using Lux
using LuxCore
using NNlib

import ..Losses: get_loss_type, GaussianMLE
import ..Models: Architecture, _broadcast_relu

include("layers.jl")

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
        push!(layers, Dense(d_in_i => d_block, relu))
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

Configuration for TabM ensemble architectures.

The chain output is 3D: `(outsize, k, batch)`. Ensemble averaging is handled
during training and at inference.

# Arguments
- `k::Int`: Number of ensemble members (default `32`).
- `n_blocks::Int`: Number of MLP blocks (default `3`).
- `d_block::Int`: Hidden dimension per block (default `512`).
- `dropout::Float64`: Dropout rate (default `0.1`).
- `arch_type::Symbol`: `:tabm`, `:tabm_mini`, or `:tabm_packed` (default `:tabm`).
- `scaling_init::Symbol`: Init for ensemble scaling — `:random_signs`, `:normal`, or `:ones`
  (default `:random_signs`). Automatically overridden to `:normal` when embeddings are used.
- `MLE_tree_split::Bool`: Split output head for Gaussian MLE (default `false`).
"""
struct TabMConfig <: Architecture
    k::Int
    n_blocks::Int
    d_block::Int
    dropout::Float64
    arch_type::Symbol
    scaling_init::Symbol
    MLE_tree_split::Bool
end

function TabMConfig(; kwargs...)
    args = Dict{Symbol,Any}(
        :k => 32,
        :n_blocks => 3,
        :d_block => 512,
        :dropout => 0.1,
        :arch_type => :tabm,
        :scaling_init => :random_signs,
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

    return TabMConfig(
        args[:k], args[:n_blocks], args[:d_block], args[:dropout],
        Symbol(args[:arch_type]), Symbol(args[:scaling_init]),
        args[:MLE_tree_split],
    )
end

function (config::TabMConfig)(; nfeats, outsize, d_features=nothing, scaling_init_override=nothing)
    @assert config.k > 0 "k must be > 0, got $(config.k)"
    @assert nfeats > 0 "nfeats must be > 0, got $nfeats"
    @assert outsize > 0 "outsize must be > 0, got $outsize"

    k = config.k
    d_block = config.d_block
    d_in = nfeats

    if isnothing(d_features)
        d_features = ones(Int, nfeats)
    end

    effective_scaling_init = isnothing(scaling_init_override) ? config.scaling_init : scaling_init_override

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

    return Chain(EnsembleView(k), bb..., head)
end

end