struct NeuroTree{M,V,F}
    w::M
    b::V
    s::V
    p::M
    ml::M
    ms::M
    actA::F
    scaler::Bool
    ntrees::Int
end
@layer NeuroTree trainable = (w, b, s, p)

function (m::NeuroTree)(x)
    if m.scaler
        nw = softplus(m.s) .* (m.actA(m.w) * x .+ m.b) # [F,B] => [NT,B]
    else
        nw = (m.actA(m.w) * x .+ m.b) # [F,B] => [NT,B]
    end
    nw = reshape(nw, size(m.ml, 2), :) # [NT,B] => [N,TB]
    lw = exp.(m.ml * nw .- m.ms * softplus.(nw)) # [N,TB] => [L,TB]
    lw = reshape(lw, :, size(x, 2)) # [L,TB] => [LT,B]
    p = m.p * lw ./ m.ntrees # [P,LT] * [LT,B] => [P,B]
    return p
end

"""
    NeuroTree(; ins, outs, , tree_type=:binary, depth=4, ntrees=64, proj_size=1, actA=identity, scaler=true, init_scale=1e-1)
    NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; tree_type=:binary, depth=4, ntrees=64, proj_size=1, actA=identity, scaler=true, init_scale=1e-1)

Initialization of a NeuroTree.
"""
function NeuroTree(; ins, outs, tree_type=:binary, depth=4, ntrees=64, proj_size=1, actA=identity, scaler=true, init_scale=1e-1)
    ml = get_logits_mask(Val(tree_type), depth)
    ms = get_softplus_mask(Val(tree_type), depth)
    nleaves = size(ml, 1)
    nnodes = size(ml, 2)

    op = NeuroTree(
        Float32.((rand(nnodes * ntrees, ins) .- 0.5) ./ 4), # w
        zeros(Float32, nnodes * ntrees), # b
        Float32.(fill(log(exp(1) - 1), nnodes * ntrees)), # s
        Float32.(randn(outs, nleaves * ntrees) .* init_scale), # p
        Float32.(ml),
        Float32.(ms),
        actA,
        scaler,
        ntrees,
    )
    return op
end
function NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; tree_type=:binary, depth=4, ntrees=64, proj_size=1, actA=identity, scaler=true, init_scale=1e-1)
    ml = get_logits_mask(Val(tree_type), depth)
    ms = get_softplus_mask(Val(tree_type), depth)
    nleaves = size(ml, 1)
    nnodes = size(ml, 2)

    op = NeuroTree(
        Float32.((rand(nnodes * ntrees, ins) .- 0.5) ./ 4), # w
        Float32.((rand(nnodes * ntrees) .- 0.5) ./ 4), # b
        Float32.(fill(log(exp(1) - 1), nnodes * ntrees)), # s
        Float32.(randn(outs, nleaves * ntrees) .* init_scale), # p
        Float32.(ml),
        Float32.(ms),
        actA,
        scaler,
        ntrees,
    )
    return op
end


"""
    get_logits_mask(::Val{:binary}, depth::Integer)
    get_logits_mask(::Val{:oblivious}, depth::Integer)

Returns a masking matrix for logits for tree leaves probabilities.
"""
function get_logits_mask(::Val{:binary}, depth::Integer)
    nodes = 2^depth - 1
    leaves = 2^depth
    mask = zeros(Bool, leaves, nodes)
    for d in 1:depth
        blocks = 2^(d - 1)
        k = 2^(depth - d)
        stride = 2 * k
        for b in 1:blocks
            view(mask, (b-1)*stride+1:(b-1)*stride+k, 2^(d - 1) + b - 1) .= 1
        end
    end
    return mask
end
function get_logits_mask(::Val{:oblivious}, depth::Integer)
    leaves = 2^depth
    mask = zeros(Bool, leaves, depth)
    for d in 1:depth
        blocks = 2^(d - 1)
        k = 2^(depth - d)
        stride = 2 * k
        for b in 1:blocks
            view(mask, (b-1)*stride+1:(b-1)*stride+k, d) .= 1
        end
    end
    return mask
end

"""
    get_softplus_mask(::Val{:binary}, depth::Integer)
    get_softplus_mask(::Val{:oblivious}, depth::Integer)

Returns a masking matrix for softplus terms for tree leaves probabilities.
"""
function get_softplus_mask(::Val{:binary}, depth::Integer)
    nodes = 2^depth - 1
    leaves = 2^depth
    mask = zeros(Bool, leaves, nodes)
    for d in 1:depth
        blocks = 2^(d - 1)
        k = 2^(depth - d + 1)
        stride = k
        for b in 1:blocks
            view(mask, (b-1)*stride+1:(b-1)*stride+k, 2^(d - 1) + b - 1) .= 1
        end
    end
    return mask
end
function get_softplus_mask(::Val{:oblivious}, depth::Integer)
    leaves = 2^depth
    mask = ones(Bool, leaves, depth)
    return mask
end

"""
    StackTree

A StackTree is made of a collection of NeuroTree.
"""
struct StackTree
    trees::Vector{NeuroTree}
end
@layer StackTree

function StackTree((ins, outs)::Pair{<:Integer,<:Integer}; tree_type=:binary, depth=4, ntrees=64, proj_size=1, stack_size=1, hidden_size=8, actA=identity, scaler=true, init_scale=1e-1)
    @assert stack_size == 1 || hidden_size >= outs
    trees = []
    for i in 1:stack_size
        if i == 1
            if i < stack_size
                tree = NeuroTree(ins => hidden_size; tree_type, depth, ntrees, proj_size, actA, scaler, init_scale)
                push!(trees, tree)
            else
                tree = NeuroTree(ins => outs; tree_type, depth, ntrees, proj_size, actA, scaler, init_scale)
                push!(trees, tree)
            end
        elseif i < stack_size
            tree = NeuroTree(hidden_size => hidden_size; tree_type, depth, ntrees, proj_size, actA, scaler, init_scale)
            push!(trees, tree)
        else
            tree = NeuroTree(hidden_size => outs; tree_type, depth, ntrees, proj_size, actA, scaler, init_scale)
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
