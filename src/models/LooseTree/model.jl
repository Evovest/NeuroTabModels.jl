struct LooseTree{F} <: AbstractLuxLayer
    tree_type::Symbol
    actA::F
    scaler::Bool
    feats::Int
    outs::Int
    depth::Int
    trees::Int
    nodes::Int
    leaves::Int
    k::Int
    init_scale::Float32
end

function LooseTree(; feats, outs, tree_type=:binary, actA=identity, scaler=true, depth, trees, k, init_scale=0.1)
    @assert tree_type ∈ [:binary, :oblivious]
    nodes = tree_type == :binary ? 2^depth - 1 : depth
    leaves = 2^depth
    return LooseTree(tree_type, actA, scaler, feats, outs, depth, trees, nodes, leaves, k, Float32(init_scale))
end
function LooseTree((feats, outs)::Pair{<:Integer,<:Integer}; tree_type=:binary, actA=identity, scaler=true, depth, trees, k, init_scale=0.1)
    @assert tree_type ∈ [:binary, :oblivious]
    nodes = tree_type == :binary ? 2^depth - 1 : depth
    leaves = 2^depth
    return LooseTree(tree_type, actA, scaler, feats, outs, depth, trees, nodes, leaves, k, Float32(init_scale))
end

# Define the Lux interface
function LuxCore.initialparameters(rng::AbstractRNG, l::LooseTree)
    return (
        w=Float32.((rand(rng, l.nodes * l.trees * l.k, l.feats) .- 0.5) ./ 4), # [NTK,F]
        b=zeros(Float32, l.nodes * l.trees * l.k), # [NTK]
        s=Float32.(fill(log(expm1(1)), l.nodes * l.trees * l.k)), # [NTK]
        p=Float32.(randn(rng, l.outs, l.leaves, l.trees) .* l.init_scale), # [P,L,T,K]
    )
end

function LuxCore.initialstates(rng::AbstractRNG, l::LooseTree)
    return (
        ml=Float32.(get_logits_mask(Val(l.tree_type), l.depth)),
        ms=Float32.(get_softplus_mask(Val(l.tree_type), l.depth)),
    )
end

function (l::LooseTree)(x, ps, st)
    if l.scaler
        nw = softplus(ps.s) .* (l.actA(ps.w) * x .+ ps.b) # [F,B] => [NTK,B]
    else
        nw = (l.actA(ps.w) * x .+ ps.b) # [F,B] => [NTK,B]
    end
    nw = reshape(nw, size(st.ml, 2), :) # [NTK,B] => [N,TKB]
    lw = exp.(st.ml * nw .- st.ms * softplus.(nw)) # [N,TKB] => [L,TKB]
    lw = reshape(lw, 1, l.leaves, l.trees, l.k, size(x, 2)) # [L,TKB] => [1,L,T,K,B]
    y1 = dropdims(sum(ps.p .* lw; dims=2); dims=2) # [P,L,T,K,T] * [1,L,T,K,B] => [P,T,K,B]
    y = dropdims(mean(y1; dims=2); dims=2) # [P,T,K,B] => [P,K,B]
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
