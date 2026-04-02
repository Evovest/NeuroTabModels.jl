using Lux
using Random
using NNlib

"""
    NLinear(n, in_features, out_features; bias=true)

A batch of `n` independent linear layers applied in parallel via `batched_mul`.
Input shape `(in_features, n, batch)` → output shape `(out_features, n, batch)`.

# Arguments
- `n::Int`: Number of independent linear layers (typically one per feature).
- `in_features::Int`: Input dimension for each linear layer.
- `out_features::Int`: Output dimension for each linear layer.
- `bias::Bool`: Whether to include a learnable bias (default `true`).
"""
struct NLinear <: Lux.AbstractLuxLayer
    n::Int
    in_features::Int
    out_features::Int
    use_bias::Bool
end

function NLinear(n::Int, in_features::Int, out_features::Int; bias::Bool=true)
    return NLinear(n, in_features, out_features, bias)
end

function Lux.initialparameters(rng::AbstractRNG, l::NLinear)
    limit = Float32(l.in_features)^(-0.5f0)
    weight = (rand(rng, Float32, l.out_features, l.in_features, l.n) .* 2f0 .* limit) .- limit

    if l.use_bias
        return (weight=weight, bias=zeros(Float32, l.out_features, 1, l.n))
    else
        return (weight=weight,)
    end
end

Lux.initialstates(::AbstractRNG, ::NLinear) = (;)

function (l::NLinear)(x::AbstractArray{T,3}, ps, st) where T
    x_perm = PermutedDimsArray(x, (1, 3, 2))
    out = batched_mul(ps.weight, x_perm)

    if l.use_bias
        out = out .+ ps.bias
    end

    return permutedims(out, (1, 3, 2)), st
end