module ResNet

export ResNetConfig

import Flux
import Flux: @functor, trainmode!, gradient, Chain, DataLoader, cpu, gpu
import Flux: logsoftmax, softmax, softmax!, relu, sigmoid, sigmoid_fast, hardsigmoid, tanh, tanh_fast, hardtanh, softplus, onecold, onehotbatch
import Flux: BatchNorm, Dense, Dropout, MultiHeadAttention, Parallel, SkipConnection

import ..Models: get_loss_type, GaussianMLE
import ..Models: Architecture

struct ResNetConfig <: Architecture
    num_blocks::Int
    hidden_size::Int
    act::Symbol
    dropout::Float64
    MLE_tree_split::Bool
end

function ResNetConfig(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :num_blocks => 1,
        :hidden_size => 64,
        :act => :relu,
        :dropout => 1.0,
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

    config = ResNetConfig(
        args[:num_blocks],
        args[:hidden_size],
        Symbol(args[:act]),
        args[:dropout],
        args[:MLE_tree_split]
    )

    return config
end

function ResBlock_v1(; hsize, dropout)
    layer = SkipConnection(
        Chain(
            BatchNorm(hsize),
            Dense(hsize => hsize, relu),
            Dropout(dropout),
            Dense(hsize => hsize),
            Dropout(dropout),
        ),
        +
    )
    return layer
end
function ResBlock_v2A(; hsize, dropout, kwargs...)
    layer = Chain(
        SkipConnection(
            Chain(
                Dense(hsize => hsize),
                BatchNorm(hsize, relu),
                Dropout(dropout),
                Dense(hsize => hsize),
                BatchNorm(hsize),
            ),
            +
        ),
        x -> relu.(x)
    )
    return layer
end
function ResBlock_v2B(; hsize, dropout, kwargs...)
    layer = Chain(
        SkipConnection(
            Chain(
                Dense(hsize => hsize),
                BatchNorm(hsize, relu),
                Dropout(dropout),
                Dense(hsize => hsize),
                BatchNorm(hsize),
            ),
            vcat
        ),
        Dense(2 * hsize => hsize, relu),
    )
    return layer
end


function (config::ResNetConfig)(; nfeats, outsize)

    hsize = config.hidden_size
    dropout = config.dropout

    if config.MLE_tree_split && outsize == 2
        outsize ÷= 2
        chain = Chain(
            BatchNorm(nfeats),
            Dense(nfeats => hsize),
            BatchNorm(hsize, relu),
            Parallel(
                vcat,
                Chain(
                    ResBlock_v2A(; hsize, dropout),
                    # BatchNorm(hsize),
                    Dense(hsize => outsize),
                    # BatchNorm(outsize),
                ),
                Chain(
                    ResBlock_v2A(; hsize, dropout),
                    # BatchNorm(hsize),
                    Dense(hsize => outsize),
                    # BatchNorm(outsize),
                )
            ),
        )
    else
        chain = Chain(
            BatchNorm(nfeats),
            Dense(nfeats => hsize),
            BatchNorm(hsize, relu),
            ResBlock_v2A(; hsize, dropout),
            # BatchNorm(hsize),
            Dense(hsize => outsize),
            # BatchNorm(outsize),
        )
        @info "single chain"
    end

    return chain
end

end