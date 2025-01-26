module NeuroTrees

import .Threads: @threads

using CUDA

import Flux
import Flux: @layer, trainmode!, gradient, Chain, DataLoader, cpu, gpu
import Flux: logσ, logsoftmax, softmax, softmax!, sigmoid, sigmoid_fast, hardsigmoid, tanh, tanh_fast, hardtanh, softplus, onecold, onehotbatch
import Flux: BatchNorm, Dense, MultiHeadAttention, Parallel

using ChainRulesCore
import ChainRulesCore: rrule

import ..Losses: get_loss_type, GaussianMLE
import ..Models: get_model_chain, ArchType

include("model.jl")

struct NeuroTreeConfig
    loss::Symbol
    nrounds::Int
    lr::Float32
    wd::Float32
    batchsize::Int
    actA::Symbol
    depth::Int
    ntrees::Int
    hidden_size::Int
    stack_size::Int
    init_scale::Float32
    MLE_tree_split::Bool
end

function NeuroTreeConfig(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :loss => :mse,
        :nrounds => 100,
        :lr => 1.0f-2,
        :wd => 0.0f0,
        :batchsize => 2048,
        :actA => :tanh,
        :depth => 4,
        :ntrees => 64,
        :hidden_size => 1,
        :stack_size => 1,
        :init_scale => 0.1,
        :MLE_tree_split => false,
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    args_ignored_str = join(args_ignored, ", ")
    length(args_ignored) > 0 &&
        @info "Following $(length(args_ignored)) provided arguments will be ignored: $(args_ignored_str)."

    args_default = setdiff(keys(args), keys(kwargs))
    args_default_str = join(args_default, ", ")
    length(args_default) > 0 &&
        @info "Following $(length(args_default)) arguments were not provided and will be set to default: $(args_default_str)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:loss] = Symbol(args[:loss])
    args[:loss] ∉ [:mse, :mae, :logloss, :gaussian_mle, :tweedie_deviance] && error("The provided kwarg `loss`: `$(args[:loss])` is not supported.")

    config = NeuroTreeConfig(
        args[:loss],
        args[:nrounds],
        Float32(args[:lr]),
        Float32(args[:wd]),
        args[:batchsize],
        Symbol(args[:actA]),
        args[:depth],
        args[:ntrees],
        args[:hidden_size],
        args[:stack_size],
        args[:init_scale],
        args[:MLE_tree_split],
    )

    return config
end


function get_model_chain(::Type{ArchType{:neurotree}}, config; nfeats, outsize, kwargs...)

    L = get_loss_type(config.loss)

    if L <: GaussianMLE && config.MLE_tree_split
        chain = Chain(
            BatchNorm(nfeats),
            Parallel(
                vcat,
                StackTree(nfeats => outsize;
                    depth=config.depth,
                    ntrees=config.ntrees,
                    stack_size=config.stack_size,
                    hidden_size=config.hidden_size,
                    actA=_act_dict[config.actA],
                    init_scale=config.init_scale),
                StackTree(nfeats => outsize;
                    depth=config.depth,
                    ntrees=config.ntrees,
                    stack_size=config.stack_size,
                    hidden_size=config.hidden_size,
                    actA=_act_dict[config.actA],
                    init_scale=config.init_scale)
            )
        )
    else
        outsize = L <: GaussianMLE ? 2 * outsize : outsize
        chain = Chain(
            BatchNorm(nfeats),
            StackTree(nfeats => outsize;
                depth=config.depth,
                ntrees=config.ntrees,
                stack_size=config.stack_size,
                hidden_size=config.hidden_size,
                actA=_act_dict[config.actA],
                init_scale=config.init_scale)
        )

    end
    return chain
end
function get_model_chain(config::NeuroTreeConfig)

    L = get_loss_type(config.loss)

    if L <: GaussianMLE && config.MLE_tree_split
        outsize = config.outsize
        chain = Chain(
            BatchNorm(config.nfeats),
            Parallel(
                vcat,
                StackTree(config.nfeats => outsize;
                    depth=config.depth,
                    ntrees=config.ntrees,
                    stack_size=config.stack_size,
                    hidden_size=config.hidden_size,
                    actA=_act_dict[config.actA],
                    init_scale=config.init_scale),
                StackTree(config.nfeats => outsize;
                    depth=config.depth,
                    ntrees=config.ntrees,
                    stack_size=config.stack_size,
                    hidden_size=config.hidden_size,
                    actA=_act_dict[config.actA],
                    init_scale=config.init_scale)
            )
        )
    else
        outsize = L <: GaussianMLE ? 2 * config.outsize : config.outsize
        chain = Chain(
            BatchNorm(config.nfeats),
            StackTree(config.nfeats => outsize;
                depth=config.depth,
                ntrees=config.ntrees,
                stack_size=config.stack_size,
                hidden_size=config.hidden_size,
                actA=_act_dict[config.actA],
                init_scale=config.init_scale)
        )

    end
    return chain
end

end