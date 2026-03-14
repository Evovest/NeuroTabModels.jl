struct NeuroTree{F} <: AbstractLuxLayer
    tree_type::Symbol
    actA::F
    scaler::Bool
    feats::Int
    outs::Int
    depth::Int
    trees::Int
    nodes::Int
    leaves::Int
    init_scale::Float32
end

function NeuroTree(; feats, outs, tree_type=:binary, actA=identity, scaler=true, depth, trees, init_scale=0.1)
    nodes = 2^depth - 1
    leaves = 2^depth
    return NeuroTree(tree_type, actA, scaler, feats, outs, depth, trees, nodes, leaves, Float32(init_scale))
end
function NeuroTree((feats, outs)::Pair{<:Integer,<:Integer}; tree_type=:binary, actA=identity, scaler=true, depth, trees, init_scale=0.1)
    nodes = 2^depth - 1
    leaves = 2^depth
    return NeuroTree(tree_type, actA, scaler, feats, outs, depth, trees, nodes, leaves, Float32(init_scale))
end

# Define the Lux interface
function LuxCore.initialparameters(rng::AbstractRNG, l::NeuroTree)
    return (
        w=Float32.((rand(rng, l.nodes * l.trees, l.feats) .- 0.5) ./ 4), # w
        b=zeros(Float32, l.nodes * l.trees), # b
        s=Float32.(fill(log(exp(1) - 1), l.nodes * l.trees)), # s
        p=Float32.(randn(rng, l.outs, l.leaves * l.trees) .* l.init_scale), # p
    )
end

function LuxCore.initialstates(rng::AbstractRNG, l::NeuroTree)
    return (
        ml=Float32.(get_logits_mask(Val(l.tree_type), l.depth)),
        ms=Float32.(get_softplus_mask(Val(l.tree_type), l.depth)),
    )
end

function (l::NeuroTree)(x, ps, st)
    if l.scaler
        nw = softplus(ps.s) .* (l.actA(ps.w) * x .+ ps.b) # [F,B] => [NT,B]
    else
        nw = (l.actA(ps.w) * x .+ ps.b) # [F,B] => [NT,B]
    end
    nw = reshape(nw, size(st.ml, 2), :) # [NT,B] => [N,TB]
    lw = exp.(st.ml * nw .- st.ms * softplus.(nw)) # [N,TB] => [L,TB]
    lw = reshape(lw, :, size(x, 2)) # [L,TB] => [LT,B]
    y = ps.p * lw ./ l.trees # [P,LT] * [LT,B] => [P,B]
    return y, st
end


"""
    get_logits_mask(::Val{:binary}, depth::Integer)
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
