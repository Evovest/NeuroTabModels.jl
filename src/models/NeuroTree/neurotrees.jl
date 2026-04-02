module NeuroTrees

export NeuroTreeConfig

using Random
using Lux
using LuxCore
using Statistics: mean
using NNlib: softplus, sigmoid_fast, hardsigmoid, tanh_fast, hardtanh, tanhshrink

import ..Losses: get_loss_type, GaussianMLE
import ..Models: Architecture

include("model.jl")

struct StackedNeuroTree{L} <: LuxCore.AbstractLuxWrapperLayer{:chain}
    chain::L
end

function StackedNeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; hidden_size::Int, stack_size::Int, k::Int=1, tree_kwargs...)
    if stack_size == 1
        return StackedNeuroTree(NeuroTree(ins => outs; k, tree_kwargs...))
    end

    layers = Any[NeuroTree(ins => 1; k=hidden_size, tree_kwargs...), FlattenLayer()]
    for _ in 2:(stack_size-1)
        push!(layers, SkipConnection(
            Chain(NeuroTree(hidden_size => 1; k=hidden_size, tree_kwargs...), FlattenLayer()), +
        ))
    end
    push!(layers, NeuroTree(hidden_size => outs; k, tree_kwargs...))

    return StackedNeuroTree(Chain(layers...))
end

struct NeuroTreeConfig <: Architecture
    tree_type::Symbol
    actA::Symbol
    depth::Int
    ntrees::Int
    k::Int
    hidden_size::Int
    stack_size::Int
    scaler::Bool
    init_scale::Float32
    MLE_tree_split::Bool
end

function NeuroTreeConfig(; kwargs...)
    args = Dict{Symbol,Any}(
        :tree_type => :binary,
        :actA => :identity,
        :depth => 4,
        :ntrees => 32,
        :k => 1,
        :hidden_size => 1,
        :stack_size => 1,
        :scaler => true,
        :init_scale => 0.1,
        :MLE_tree_split => false,
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    length(args_ignored) > 0 &&
        @warn "Following $(length(args_ignored)) provided arguments will be ignored: $(join(args_ignored, ", "))."

    args_default = setdiff(keys(args), keys(kwargs))
    length(args_default) > 0 &&
        @info "Following $(length(args_default)) arguments were not provided and will be set to default: $(join(args_default, ", "))."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    return NeuroTreeConfig(
        Symbol(args[:tree_type]),
        Symbol(args[:actA]),
        args[:depth],
        args[:ntrees],
        args[:k],
        args[:hidden_size],
        args[:stack_size],
        args[:scaler],
        args[:init_scale],
        args[:MLE_tree_split],
    )
end

function _tree_kwargs(config::NeuroTreeConfig)
    return (;
        config.tree_type,
        config.depth,
        trees=config.ntrees,
        # k=config.k,
        actA=act_dict[config.actA],
        config.scaler,
        config.init_scale,
    )
end

function (config::NeuroTreeConfig)(; nfeats, outsize, kwargs...)
    kwargs = _tree_kwargs(config)

    if config.MLE_tree_split
        iseven(outsize) || error("MLE_tree_split requires an even `outsize` (e.g., 2 for μ and σ). Got: $outsize")
        head_outsize = outsize ÷ 2
        chain = Chain(
            Parallel(
                vcat,
                StackedNeuroTree(nfeats => head_outsize; config.hidden_size, config.stack_size, config.k, kwargs...),
                StackedNeuroTree(nfeats => head_outsize; config.hidden_size, config.stack_size, config.k, kwargs...),
            ),
        )
    else
        chain = Chain(
            StackedNeuroTree(nfeats => outsize; config.hidden_size, config.stack_size, config.k, kwargs...),
        )
    end

    return chain
end

function _identity_act(x)
    return x ./ sum(abs.(x), dims=2)
end
function _tanh_act(x)
    x = tanh_fast.(x)
    return x ./ sum(abs.(x), dims=2)
end
function _hardtanh_act(x)
    x = hardtanh.(x)
    return x ./ sum(abs.(x), dims=2)
end
function _tanhshrink_act(x)
    x = tanhshrink.(x)
    return x ./ sum(abs.(x), dims=2)
end

"""
    act_dict = Dict(
        :identity => _identity_act,
        :tanh => _tanh_act,
        :hardtanh => _hardtanh_act,
        :tanhshrink => _tanhshrink_act,
    )
Dictionary mapping features activation name to their function.
"""
const act_dict = Dict(
    :identity => _identity_act,
    :tanh => _tanh_act,
    :hardtanh => _hardtanh_act,
    :tanhshrink => _tanhshrink_act,
)

end