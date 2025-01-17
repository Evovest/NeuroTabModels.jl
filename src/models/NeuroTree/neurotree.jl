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
import ..Models: get_model_chain, ModelType

include("leaf_weights.jl")

struct NeuroTree{W,B,P,F<:Function}
    w::W
    b::B
    p::P
    actA::F
end
@layer NeuroTree
# Flux.trainable(m::NeuroTree) = (w=m.w, b=m.b, p=m.p)

function node_weights(m::NeuroTree, x)
    # [N X T, F] * [F, B] => [N x T, B]
    # nw = sigmoid_fast.(m.w * x .+ m.b)
    nw = sigmoid_fast.(m.actA.(m.w) * x .+ m.b)
    # [N x T, B] -> [N, T, B]
    return reshape(nw, :, size(m.p, 3), size(x, 2))
end

function (m::NeuroTree{W,B,P,F})(x::W) where {W,B,P,F}
    # [F, B] -> [N, T, B]
    nw = node_weights(m, x)
    # [N, T, B] -> [L, T, B]
    (_, lw) = leaf_weights!(nw)
    # [L, T, B], [P, L, T] -> [P, B]
    pred = dot_prod_agg(lw, m.p) ./ size(m.p, 3)
    return pred
end

dot_prod_agg(lw, p) = dropdims(sum(reshape(lw, 1, size(lw)...) .* p, dims=(2, 3)), dims=(2, 3))

"""
    NeuroTree(; ins, outs, depth=4, ntrees=64, actA=identity, init_scale=1.0)
    NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, actA=identity, init_scale=1.0)

Initialization of a NeuroTree.
"""
function NeuroTree(; ins, outs, depth=4, ntrees=64, actA=identity, init_scale=1.0)
    nnodes = 2^depth - 1
    nleaves = 2^depth
    nt = NeuroTree(
        Flux.glorot_uniform(nnodes * ntrees, ins), # w
        zeros(Float32, nnodes * ntrees), # b
        Float32.((rand(outs, nleaves, ntrees) .- 0.5) .* sqrt(12) .* init_scale), # p
        # Float32.(randn(outs, nleaves, ntrees) ./ 1 .* init_scale), # p
        actA,
    )
    return nt
end
function NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, actA=identity, init_scale=1.0)
    nnodes = 2^depth - 1
    nleaves = 2^depth
    nt = NeuroTree(
        Flux.glorot_uniform(nnodes * ntrees, ins), # w
        zeros(Float32, nnodes * ntrees), # b
        Float32.((rand(outs, nleaves, ntrees) .- 0.5) .* sqrt(12) .* init_scale), # p
        # Float32.(randn(outs, nleaves, ntrees) ./ 1 .* init_scale), # p
        actA,
    )
    return nt
end

"""
    StackTree
A StackTree is made of a collection of NeuroTree.
"""
struct StackTree
    trees::Vector{NeuroTree}
end
@layer StackTree

function StackTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, stack_size=2, hidden_size=8, actA=identity, init_scale=1.0)
    @assert stack_size == 1 || hidden_size >= outs
    trees = []
    for i in 1:stack_size
        if i == 1
            if i < stack_size
                tree = NeuroTree(ins => hidden_size; depth, ntrees, actA, init_scale)
                push!(trees, tree)
            else
                tree = NeuroTree(ins => outs; depth, ntrees, actA, init_scale)
                push!(trees, tree)
            end
        elseif i < stack_size
            tree = NeuroTree(hidden_size => hidden_size; depth, ntrees, actA, init_scale)
            push!(trees, tree)
        else
            tree = NeuroTree(hidden_size => outs; depth, ntrees, actA, init_scale)
            push!(trees, tree)
        end
    end
    m = StackTree(trees)
    return m
end

function (m::StackTree)(x::AbstractMatrix)
    p = m.trees[1](x)
    for i in 2:length(m.trees)
        if i < length(m.trees)
            p = p .+ m.trees[i](p)
        else
            _p = m.trees[i](p)
            p = view(p, 1:size(_p, 1), :) .+ _p
        end
    end
    return p
end
# function (m::StackTree)(x::AbstractMatrix)
#     p = m.trees[1](x)
#     for i in 2:length(m.trees)
#         p = m.trees[i](p)
#     end
#     return p
# end

const _act_dict = Dict(
    :identity => identity,
    :tanh => tanh,
    :hardtanh => hardtanh,
    :sigmoid => sigmoid,
    :hardsigmoid => hardsigmoid
)

function get_model_chain(::Type{ModelType{:neurotree}}, config; nfeats, outsize, kwargs...)

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

end