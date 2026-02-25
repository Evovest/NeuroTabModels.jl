module MLP

export MLPConfig

import Flux
import Flux: @functor, trainmode!, gradient, Chain, DataLoader, cpu, gpu
import Flux: logsoftmax, softmax, softmax!, relu, sigmoid, sigmoid_fast, hardsigmoid, tanh, tanh_fast, hardtanh, softplus, onecold, onehotbatch
import Flux: BatchNorm, Dense, Dropout, MultiHeadAttention, Parallel, SkipConnection

import ..Models: get_loss_type, GaussianMLE
import ..Models: Architecture

struct MLPConfig <: Architecture
    act::Symbol
    hidden_size::Int
    stack_size::Int
    MLE_tree_split::Bool
end

function MLPConfig(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :act => :relu,
        :hidden_size => 64,
        :stack_size => 1,
        :MLE_tree_split => false
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

    config = MLPConfig(
        Symbol(args[:act]),
        args[:hidden_size],
        args[:stack_size],
        args[:MLE_tree_split]
    )

    return config
end

function (config::MLPConfig)(; nfeats, outsize)

    hsize = config.hidden_size

    if config.MLE_tree_split && outsize == 2
        outsize ÷= 2
        chain = Chain(
            BatchNorm(nfeats),
            Dense(nfeats => hsize),
            Parallel(
                vcat,
                Chain(
                    BatchNorm(nfeats),
                    Dense(nfeats => hsize),
                    BatchNorm(hsize, relu),
                    Dense(hsize => outsize)
                ),
                Chain(
                    BatchNorm(nfeats),
                    Dense(nfeats => hsize),
                    BatchNorm(hsize, relu),
                    Dense(hsize => outsize)
                )
            )
        )
    else
        chain = Chain(
            BatchNorm(nfeats),
            Dense(nfeats => hsize),
            BatchNorm(hsize, relu),
            Dense(hsize => hsize),
            BatchNorm(hsize, relu),
            Dense(hsize => outsize)
        )
    end

    return chain
end

end