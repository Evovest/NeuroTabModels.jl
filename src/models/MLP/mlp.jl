module MLP

export MLPConfig

using Lux
using LuxCore

import ..Models: Architecture, get_activation

"""
    MLPConfig(; kwargs...)

Configuration for a multi-layer perceptron backbone.

# Arguments
- `act::Symbol`: Activation — `:relu`, `:gelu`, `:sigmoid`, or `:tanh` (default `:relu`).
- `hidden_size::Int`: Hidden dimension (default `64`).
- `stack_size::Int`: Number of hidden blocks (default `1`).
- `dropout::Float64`: Dropout rate between blocks (default `0.0`).
- `MLE_tree_split::Bool`: Split output head for Gaussian MLE (default `false`).
"""
struct MLPConfig <: Architecture
    act::Symbol
    hidden_size::Int
    stack_size::Int
    dropout::Float64
    MLE_tree_split::Bool
end

function MLPConfig(; kwargs...)
    args = Dict{Symbol,Any}(
        :act => :relu,
        :hidden_size => 64,
        :stack_size => 1,
        :dropout => 0.0,
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

    return MLPConfig(
        Symbol(args[:act]),
        args[:hidden_size],
        args[:stack_size],
        args[:dropout],
        args[:MLE_tree_split],
    )
end

function _mlp_trunk(ins::Int, hsize::Int, outsize::Int, act, stack_size::Int, dropout::Float64)
    layers = Any[
        Dense(ins => hsize),
    ]
    for _ in 1:stack_size
        push!(layers, BatchNorm(hsize, act))
        push!(layers, Dense(hsize => hsize))
        dropout > 0 && push!(layers, Dropout(dropout))
    end
    push!(layers, Dense(hsize => outsize))
    return Chain(layers...)
end

"""
    (config::MLPConfig)(; ins, outsize)

Build a `Lux.Chain` from `config`.

# Arguments
- `ins::Int`: Number of input features.
- `outsize::Int`: Number of output units.

# Returns
A `Lux.Chain` of `Dense` → `BatchNorm` → `Dense` blocks with optional dropout.
"""
function (config::MLPConfig)(; ins, outsize, kwargs...)
    act = get_activation(config.act)
    hsize = config.hidden_size

    if config.MLE_tree_split
        iseven(outsize) || error("MLE_tree_split requires an even `outsize` (e.g., 2 for μ and σ). Got: $outsize")
        head_outsize = outsize ÷ 2
        chain = Chain(
            Parallel(
                vcat,
                _mlp_trunk(ins, hsize, head_outsize, act, config.stack_size, config.dropout),
                _mlp_trunk(ins, hsize, head_outsize, act, config.stack_size, config.dropout),
            ),
        )
    else
        chain = _mlp_trunk(ins, hsize, outsize, act, config.stack_size, config.dropout)
    end

    return chain
end

end
