module TabM

export TabMConfig

using Lux
using LuxCore
using Random: AbstractRNG, rand, randn

import ..Models: Architecture, _broadcast_relu
using ..GroupedDenseLayer: GroupedDense, rsqrt_uniform_grouped

include("layers.jl")

function _batch_ensemble_backbone(; d_in::Int, n_blocks::Int, d_block::Int, dropout::Float64, k::Int)
    layers = Any[]
    for i in 1:n_blocks
        d_in_i = (i == 1) ? d_in : d_block
        if i == 1
            push!(layers, LinearBatchEnsemble(d_in_i, d_block; k, scaling_init=(:normal, :ones)))
        else
            push!(layers, LinearBatchEnsemble(d_in_i, d_block; k, scaling_init=:ones))
        end
        push!(layers, WrappedFunction(_broadcast_relu))
        dropout > 0 && push!(layers, Dropout(dropout))
    end
    return layers
end

function _mini_ensemble_backbone(; d_in::Int, n_blocks::Int, d_block::Int, dropout::Float64, k::Int)
    layers = Any[ScaleEnsemble(k, d_in; init=:normal, bias=false)]
    for i in 1:n_blocks
        d_in_i = (i == 1) ? d_in : d_block
        push!(layers, Dense(d_in_i => d_block, relu))
        dropout > 0 && push!(layers, Dropout(dropout))
    end
    return layers
end

function _packed_ensemble_backbone(; d_in::Int, n_blocks::Int, d_block::Int, dropout::Float64, k::Int)
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

Configuration for TabM ensemble backbones.

# Arguments
- `k::Int`: Ensemble size (default `32`).
- `n_blocks::Int`: Number of MLP blocks (default `3`).
- `d_block::Int`: Hidden dimension (default `512`).
- `dropout::Float64`: Dropout rate (default `0.1`).
- `arch_type::Symbol`: `:tabm`, `:tabm_mini`, or `:tabm_packed` (default `:tabm`).
- `MLE_tree_split::Bool`: Split output head for Gaussian MLE (default `false`).
"""
struct TabMConfig <: Architecture
    k::Int
    n_blocks::Int
    d_block::Int
    dropout::Float64
    arch_type::Symbol
    MLE_tree_split::Bool
end

function TabMConfig(; kwargs...)
    args = Dict{Symbol,Any}(
        :k => 32, :n_blocks => 3, :d_block => 512, :dropout => 0.1, :arch_type => :tabm, :MLE_tree_split => false
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
        args[:k], args[:n_blocks], args[:d_block], args[:dropout], Symbol(args[:arch_type]), args[:MLE_tree_split]
    )
end

"""
    (config::TabMConfig)(; nfeats, outsize)

Build a `Lux.Chain` from `config`. Output shape is `(outsize, k, batch)`.

# Arguments
- `nfeats::Int`: Number of input features.
- `outsize::Int`: Number of output units.
"""
function (config::TabMConfig)(; ins, outsize, kwargs...)
    @assert config.k > 0 "k must be > 0, got $(config.k)"
    @assert ins > 0 "ins must be > 0, got $ins"
    @assert outsize > 0 "outsize must be > 0, got $outsize"

    k = config.k
    d_block = config.d_block
    d_in = ins

    bb = if config.arch_type == :tabm
        _batch_ensemble_backbone(; d_in, n_blocks=config.n_blocks, d_block, dropout=config.dropout, k)
    elseif config.arch_type == :tabm_mini
        _mini_ensemble_backbone(; d_in, n_blocks=config.n_blocks, d_block, dropout=config.dropout, k)
    elseif config.arch_type == :tabm_packed
        _packed_ensemble_backbone(; d_in, n_blocks=config.n_blocks, d_block, dropout=config.dropout, k)
    else
        error("Unknown arch_type: $(config.arch_type)")
    end

    head = if config.MLE_tree_split
        iseven(outsize) || error("MLE_tree_split requires an even `outsize` (e.g., 2 for μ and σ). Got: $outsize")
        head_outsize = outsize ÷ 2
        Parallel(vcat, LinearEnsemble(d_block, head_outsize, k), LinearEnsemble(d_block, head_outsize, k))
    else
        LinearEnsemble(d_block, outsize, k)
    end

    return Chain(EnsembleView(k), bb..., head)
end

end