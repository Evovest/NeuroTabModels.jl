# include("leaf_weights.jl")
include("leaf_weights_v2-auto.jl")

struct NeuroTree{W,B,P,F<:Function}
    w::W
    s::B
    b::B
    p::P
    actA::F
    scaler::Bool
end
@layer NeuroTree

function node_weights(m::NeuroTree, x)
    # [N X T, F] * [F, B] => [N x T, B]
    if m.scaler
        nw = Flux.sigmoid_fast.(Flux.softplus.(m.s) .* (m.actA(m.w) * x .+ m.b))
    else
        nw = Flux.sigmoid_fast.(m.actA(m.w) * x .+ m.b)
    end
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
    NeuroTree(; ins, outs, depth=4, ntrees=64, actA=identity, scaler=true, init_scale=1e-1)
    NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, actA=identity, scaler=true, init_scale=1e-1)

Initialization of a NeuroTree.
"""
function NeuroTree(; ins, outs, depth=4, ntrees=64, actA=identity, scaler=true, init_scale=1e-1)
    nnodes = 2^depth - 1
    nleaves = 2^depth
    nt = NeuroTree(
        Float32.((rand(nnodes * ntrees, ins) .- 0.5) ./ 4), # w
        Float32.(fill(log(exp(1) - 1), nnodes * ntrees)), # s
        Float32.((rand(nnodes * ntrees) .- 0.5) ./ 4), # b
        Float32.(randn(outs, nleaves, ntrees) .* init_scale), # p
        actA,
        scaler
    )
    return nt
end
function NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, actA=identity, scaler=true, init_scale=1e-1)
    nnodes = 2^depth - 1
    nleaves = 2^depth
    nt = NeuroTree(
        Float32.((rand(nnodes * ntrees, ins) .- 0.5) ./ 4), # w
        Float32.(fill(log(exp(1) - 1), nnodes * ntrees)), # s
        Float32.((rand(nnodes * ntrees) .- 0.5) ./ 4), # b
        Float32.(randn(outs, nleaves, ntrees) .* init_scale), # p
        actA,
        scaler
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

function StackTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, stack_size=2, hidden_size=8, actA=identity, scaler=true, init_scale=1e-1)
    @assert stack_size == 1 || hidden_size >= outs
    trees = []
    for i in 1:stack_size
        if i == 1
            if i < stack_size
                tree = NeuroTree(ins => hidden_size; depth, ntrees, actA, scaler, init_scale)
                push!(trees, tree)
            else
                tree = NeuroTree(ins => outs; depth, ntrees, actA, scaler, init_scale)
                push!(trees, tree)
            end
        elseif i < stack_size
            tree = NeuroTree(hidden_size => hidden_size; depth, ntrees, actA, scaler, init_scale)
            push!(trees, tree)
        else
            tree = NeuroTree(hidden_size => outs; depth, ntrees, actA, scaler, init_scale)
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
