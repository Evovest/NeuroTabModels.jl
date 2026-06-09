module ResNet

export ResNetConfig

using Lux
using LuxCore

import ..Models: Architecture, get_activation

struct ResNetConfig <: Architecture
    stack_size::Int
    hidden_size::Int
    act::Symbol
    dropout::Float64
    MLE_tree_split::Bool
end

function ResNetConfig(; kwargs...)
    args = Dict{Symbol,Any}(
        :stack_size => 1,
        :hidden_size => 64,
        :act => :relu,
        :dropout => 0.0,
        :MLE_tree_split => false,
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    length(args_ignored) > 0 &&
        @warn "Following $(length(args_ignored)) provided arguments will be ignored: $(join(args_ignored, ", "))."

    args_default = setdiff(keys(args), keys(kwargs))
    length(args_default) > 0 &&
        @info "Following $(length(args_default)) arguments were not provided and will be set to default: $(join(args_default, ", "))."

    for arg in intersect(keys(args), keys(kwargs))
        args[arg] = kwargs[arg]
    end

    return ResNetConfig(
        args[:stack_size],
        args[:hidden_size],
        Symbol(args[:act]),
        args[:dropout],
        args[:MLE_tree_split],
    )
end

function _res_block(hsize::Int, act, dropout::Float64)
    layers = Any[
        Dense(hsize => hsize),
        BatchNorm(hsize, act),
    ]
    dropout > 0 && push!(layers, Dropout(dropout))
    push!(layers, Dense(hsize => hsize))
    push!(layers, BatchNorm(hsize))

    return Chain(
        SkipConnection(Chain(layers...), +),
        BatchNorm(hsize, act),
    )
end

function _resnet_trunk(nfeats::Int, hsize::Int, outsize::Int, act, stack_size::Int, dropout::Float64)
    layers = Any[
        Dense(nfeats => hsize),
        BatchNorm(hsize, act),
    ]
    for _ in 1:stack_size
        push!(layers, _res_block(hsize, act, dropout))
    end
    push!(layers, Dense(hsize => outsize))
    return Chain(layers...)
end

function (config::ResNetConfig)(; nfeats, outsize, kwargs...)
    act = get_activation(config.act)
    hsize = config.hidden_size

    if config.MLE_tree_split
        iseven(outsize) || error("MLE_tree_split requires an even `outsize` (e.g., 2 for μ and σ). Got: $outsize")
        head_outsize = outsize ÷ 2
        chain = Chain(
            Parallel(
                vcat,
                _resnet_trunk(nfeats, hsize, head_outsize, act, config.stack_size, config.dropout),
                _resnet_trunk(nfeats, hsize, head_outsize, act, config.stack_size, config.dropout),
            ),
        )
    else
        chain = _resnet_trunk(nfeats, hsize, outsize, act, config.stack_size, config.dropout)
    end

    return chain
end

end
