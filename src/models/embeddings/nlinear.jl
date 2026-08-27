"""
    NLinear(n, in_features, out_features; bias=true)

`n` independent linear layers on `(in_features, n, batch)` via `batched_mul`.

# Arguments
- `n::Int`: Number of parallel linear layers.
- `in_features::Int`: Input dimension per layer.
- `out_features::Int`: Output dimension per layer.
- `bias::Bool`: Include bias (default `true`).
"""
struct NLinear <: LuxCore.AbstractLuxLayer
    n::Int
    in_features::Int
    out_features::Int
    use_bias::Bool
end

function NLinear(n::Int, in_features::Int, out_features::Int; bias::Bool=true)
    return NLinear(n, in_features, out_features, bias)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::NLinear)
    limit = Float32(l.in_features)^(-0.5f0)
    weight = (rand(rng, Float32, l.out_features, l.in_features, l.n) .* 2.0f0 .* limit) .- limit

    if l.use_bias
        return (weight=weight, bias=zeros(Float32, l.out_features, 1, l.n))
    else
        return (weight=weight,)
    end
end

LuxCore.initialstates(::AbstractRNG, ::NLinear) = (;)

function (l::NLinear)(x::AbstractArray{T,3}, ps, st) where {T}
    x_perm = PermutedDimsArray(x, (1, 3, 2))
    out = batched_mul(ps.weight, x_perm)

    if l.use_bias
        out = out .+ ps.bias
    end

    return permutedims(out, (1, 3, 2)), st
end
