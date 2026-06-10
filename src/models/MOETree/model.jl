struct MOETree{F} <: AbstractLuxLayer
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

function MOETree((feats, outs)::Pair{<:Integer,<:Integer}; tree_type=:binary, actA=identity, scaler=true, depth, trees, k, init_scale=0.1)
    @assert tree_type ∈ [:binary, :oblivious]
    nodes = tree_type == :binary ? 2^depth - 1 : depth
    leaves = 2^depth
    return MOETree(tree_type, actA, scaler, feats, outs, depth, trees, nodes, leaves, k, Float32(init_scale))
end

# Define the Lux interface
function LuxCore.initialparameters(rng::AbstractRNG, l::MOETree)
    return (
        # router
        lw=Float32.((rand(rng, l.nodes * l.trees, l.feats) .- 0.5) ./ 4), # [NTK,F]
        lb=zeros(Float32, l.nodes * l.trees), # [NTK]
        ls=Float32.(fill(log(expm1(1)), l.nodes * l.trees)), # [NTK]
        # expert-model
        w=Float32.((rand(rng, l.nodes * l.trees * l.leaves, l.feats) .- 0.5) ./ 4), # [NTK,F]
        b=zeros(Float32, l.nodes * l.trees * l.leaves), # [NTK]
        s=Float32.(fill(log(expm1(1)), l.nodes * l.trees * l.leaves)), # [NTK]
        p=Float32.(randn(rng, l.outs, l.leaves, l.trees, l.leaves) .* l.init_scale), # [P,L,T,K]
    )
end
function LuxCore.initialstates(rng::AbstractRNG, l::MOETree)
    return (
        ml=Float32.(get_logits_mask(Val(l.tree_type), l.depth)),
        ms=Float32.(get_softplus_mask(Val(l.tree_type), l.depth)),
    )
end

function (l::MOETree)(x, ps, st)

    enw1 = softplus(ps.ls) .* (ps.lw * x .+ ps.lb) # [F,B] => [NT,B]
    enw = reshape(enw1, size(st.ml, 2), :) # [NTK,B] => [N,TB]
    lwe1 = exp.(st.ml * enw .- st.ms * softplus.(enw)) # [N,TB] => [L,TB]
    lwe2 = reshape(lwe1, 1, l.leaves, l.trees, size(x, 2)) # [L,TB] => [1,L,T,B]
    lwe = dropdims(mean(lwe2; dims=3); dims=3) # [1,L,T,B] => [1,L,B]

    nw1 = softplus(ps.s) .* (ps.w * x .+ ps.b) # [F,B] => [NTL,B]
    nw = reshape(nw1, size(st.ml, 2), :) # [NTK,B] => [N,TLB]
    lw1 = exp.(st.ml * nw .- st.ms * softplus.(nw)) # [N,TLB] => [L,TLB]
    lw = reshape(lw1, 1, l.leaves, l.trees, l.leaves, size(x, 2)) # [L,TLB] => [1,L,T,L,B]
    y1 = dropdims(sum(ps.p .* lw; dims=2); dims=2) # [P,L,T,L] .* [1,L,T,L,B] => [P,T,L,B]
    y2 = dropdims(mean(y1; dims=2); dims=2) # [P,T,L,B] => [P,L,B]

    y = sum(lwe .* y2; dims=2) # [1,L,B] .* [P,L,B] => [P,1,B]

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
