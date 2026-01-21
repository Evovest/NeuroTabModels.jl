struct NeuroTree{DI,DP,M,P}
    d_in::DI
    d_proj::DP
    mask::M
    p::P
end
@layer NeuroTree trainable = (d_in, d_proj, p)

function (m::NeuroTree)(x)
    h = m.d_in(x) # [F,B] => [HNT,B]
    h = reshape(h, size(m.d_proj.weight, 2), :) # [HNT,B] => [H,NTB]
    nw = m.d_proj(h) # [H,NTB] => [1,NTB]
    nw = reshape(nw, size(m.mask, 1), :) # [1,NTB] => [N,TB]
    lw = softmax(m.mask' * nw) # [N,TB] => [L,TB]
    lw = reshape(lw, :, size(x, 2)) # [L,TB] => [LT,B]
    p = m.p * lw ./ size(m.mask, 2) # [LT,B] => [P,B]
    return p
end
# function (m::NeuroTree)(x)
#     nw = m.d_in(x) # [F,B] => [NT,B]
#     nw = reshape(nw, size(m.mask, 1), :) # [NT,B] => [N,TB]
#     lw = softmax(m.mask' * nw) # [N,TB] => [L,TB]
#     lw = reshape(lw, :, size(x, 2)) # [L,TB] => [LT,B]
#     p = m.p * lw ./ size(m.mask, 2) # [LT,B] => [P,B]
#     return p
# end

"""
    NeuroTree(; ins, outs, depth=4, ntrees=64, actA=identity, init_scale=1e-1)
    NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, actA=identity, init_scale=1e-1)

Initialization of a NeuroTree.
"""
function NeuroTree(; ins, outs, tree_type=:binary, depth=4, ntrees=64, proj_size=1, actA=identity, scaler=true, init_scale=1e-1)
    mask = get_mask(Val(tree_type), depth)
    nnodes = size(mask, 1)
    nleaves = size(mask, 2)

    op = NeuroTree(
        Dense(ins => proj_size * nnodes * ntrees, relu), # w
        # Dense(ins => nnodes * ntrees), # w
        Dense(proj_size => 1), # s
        mask,
        Float32.(randn(outs, nleaves * ntrees) .* init_scale), # p
    )
    return op
end
function NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; tree_type=:binary, depth=4, ntrees=64, proj_size=1, actA=identity, scaler=true, init_scale=1e-1)
    mask = get_mask(Val(tree_type), depth)
    nnodes = size(mask, 1)
    nleaves = size(mask, 2)

    op = NeuroTree(
        Dense(ins => proj_size * nnodes * ntrees, relu), # w
        # Dense(ins => nnodes * ntrees), # w
        Dense(proj_size => 1), # s
        mask,
        Float32.(randn(outs, nleaves * ntrees) .* init_scale), # p
    )
    return op
end

function get_mask(::Val{:binary}, depth::Integer)
    nodes = 2^depth - 1
    leaves = 2^depth
    mask = zeros(Bool, nodes, leaves)

    for d in 1:depth
        blocks = 2^(d - 1)
        k = 2^(depth - d)
        stride = 2 * k
        for b in 1:blocks
            view(mask, 2^(d - 1) + b - 1, (b-1)*stride+1:(b-1)*stride+k,) .= true
        end
    end
    return mask
end

function get_mask(::Val{:oblivious}, depth::Integer)
    leaves = 2^depth
    mask = zeros(Bool, depth, leaves)

    for d in 1:depth
        blocks = 2^(d - 1)
        k = 2^(depth - d)
        stride = 2 * k
        for b in 1:blocks
            view(mask, d, (b-1)*stride+1:(b-1)*stride+k,) .= true
        end
    end
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
