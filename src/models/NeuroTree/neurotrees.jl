module NeuroTrees

export NeuroTreeConfig

import .Threads: @threads
using CUDA

import Flux
import Flux: @layer, trainmode!, gradient, Chain, DataLoader, cpu, gpu
import Flux: relu, logsoftmax, softmax, softmax!, sigmoid, sigmoid_fast, hardsigmoid, tanh, tanh_fast, hardtanh, softplus, onecold, onehotbatch, glorot_uniform
import Flux: BatchNorm, Dense, Dropout, MultiHeadAttention, Parallel

import ..Losses: get_loss_type, GaussianMLE
import ..Models: Architecture

include("model.jl")

struct NeuroTreeConfig <: Architecture
    tree_type::Symbol
    actA::Symbol
    depth::Int
    ntrees::Int
    proj_size::Int
    hidden_size::Int
    stack_size::Int
    scaler::Bool
    init_scale::Float32
    MLE_tree_split::Bool
end

function NeuroTreeConfig(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :tree_type => :binary,
        :actA => :identity,
        :depth => 4,
        :ntrees => 32,
        :proj_size => 1,
        :hidden_size => 1,
        :stack_size => 1,
        :scaler => true,
        :init_scale => 0.1,
        :MLE_tree_split => false,
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    args_ignored_str = join(args_ignored, ", ")
    length(args_ignored) > 0 &&
        @warn "Following $(length(args_ignored)) provided arguments will be ignored: $(args_ignored_str)."

    args_default = setdiff(keys(args), keys(kwargs))
    args_default_str = join(args_default, ", ")
    length(args_default) > 0 &&
        @info "Following $(length(args_default)) arguments were not provided and will be set to default: $(args_default_str)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    config = NeuroTreeConfig(
        Symbol(args[:tree_type]),
        Symbol(args[:actA]),
        args[:depth],
        args[:ntrees],
        args[:proj_size],
        args[:hidden_size],
        args[:stack_size],
        args[:scaler],
        args[:init_scale],
        args[:MLE_tree_split],
    )

    return config
end

function (config::NeuroTreeConfig)(; nfeats, outsize)

    if config.MLE_tree_split && outsize == 2
        outsize ÷= 2
        chain = Chain(
            BatchNorm(nfeats),
            Parallel(
                vcat,
                StackTree(nfeats => outsize;
                    tree_type=config.tree_type,
                    depth=config.depth,
                    ntrees=config.ntrees,
                    proj_size=config.proj_size,
                    stack_size=config.stack_size,
                    hidden_size=config.hidden_size,
                    actA=act_dict[config.actA],
                    scaler=config.scaler,
                    init_scale=config.init_scale),
                StackTree(nfeats => outsize;
                    tree_type=config.tree_type,
                    depth=config.depth,
                    ntrees=config.ntrees,
                    proj_size=config.proj_size,
                    stack_size=config.stack_size,
                    hidden_size=config.hidden_size,
                    actA=act_dict[config.actA],
                    scaler=config.scaler,
                    init_scale=config.init_scale)
            )
        )
    else
        chain = Chain(
            BatchNorm(nfeats),
            StackTree(nfeats => outsize;
                tree_type=config.tree_type,
                depth=config.depth,
                ntrees=config.ntrees,
                proj_size=config.proj_size,
                stack_size=config.stack_size,
                hidden_size=config.hidden_size,
                actA=act_dict[config.actA],
                scaler=config.scaler,
                init_scale=config.init_scale)
        )

    end
end


function _identity_act(x)
    return x ./ sum(abs.(x), dims=2)
end
function _tanh_act(x)
    x = Flux.tanh_fast.(x)
    return x ./ sum(abs.(x), dims=2)
end
function _hardtanh_act(x)
    x = Flux.hardtanh.(x)
    return x ./ sum(abs.(x), dims=2)
end

"""
    act_dict = Dict(
        :identity => _identity_act,
        :tanh => _tanh_act,
        :hardtanh => _hardtanh_act,
    )

Dictionary mapping features activation name to their function.
"""
const act_dict = Dict(
    :identity => _identity_act,
    :tanh => _tanh_act,
    :hardtanh => _hardtanh_act,
)

end