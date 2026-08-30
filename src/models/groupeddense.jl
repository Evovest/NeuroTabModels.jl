module GroupedDenseLayer

export GroupedDense, rsqrt_uniform_grouped, glorot_uniform_grouped

using LuxCore
using Lux: zeros32
using LuxLib: batched_matmul
using Random: AbstractRNG

"""
    rsqrt_uniform_grouped(rng, out, in, groups)

Independent uniform init on `[-1/√in, 1/√in]` for each group. Matches the previous
`NLinear` / packed-TabM weight init; `glorot_uniform` on a 3D array would use conv-style
`nfan` and fold `groups` into the fan.
"""
function rsqrt_uniform_grouped(rng::AbstractRNG, out::Integer, in::Integer, groups::Integer)
    s = Float32(1 / sqrt(in))
    return s .* (2.0f0 .* rand(rng, Float32, out, in, groups) .- 1.0f0)
end

"""
    glorot_uniform_grouped(rng, out, in, groups)

Glorot/Xavier uniform using the 2D fan of each group (`out × in`), not conv `nfan`.
"""
function glorot_uniform_grouped(rng::AbstractRNG, out::Integer, in::Integer, groups::Integer)
    scale = Float32(sqrt(24 / (in + out)))
    return (rand(rng, Float32, out, in, groups) .- 0.5f0) .* scale
end

"""
    GroupedDense(in_dims => out_dims, n_groups, activation=identity; kwargs...)

Independent dense map `in_dims → out_dims` for each of `n_groups` groups.

# Shapes
- Input: `(in_dims, n_groups, batch)`
- Weight: `(out_dims, in_dims, n_groups)`
- Bias: `(out_dims, n_groups, 1)`
- Output: `(out_dims, n_groups, batch)`

Use this for packed MLP ensembles (`n_groups = k`) and for per-feature numerical
embeddings (`n_groups = nfeats`). Shared-weight batch ensemble (`LinearBatchEnsemble`)
is a different operator.

# Keyword arguments
- `use_bias::Bool`: Include bias (default `true`).
- `init_weight`: Called as `init_weight(rng, out_dims, in_dims, n_groups)`. Default
  `rsqrt_uniform_grouped` (fan-in of each group).
- `init_bias`: Called as `init_bias(rng, out_dims, n_groups, 1)`. Default `zeros32`.
"""
struct GroupedDense{F,IW,IB} <: LuxCore.AbstractLuxLayer
    activation::F
    in_dims::Int
    out_dims::Int
    n_groups::Int
    init_weight::IW
    init_bias::IB
    use_bias::Bool
end

function GroupedDense(
    in_dims::Int,
    out_dims::Int,
    n_groups::Int,
    activation=identity;
    use_bias::Bool=true,
    init_weight=rsqrt_uniform_grouped,
    init_bias=zeros32,
)
    in_dims > 0 || throw(ArgumentError("in_dims must be > 0, got $in_dims"))
    out_dims > 0 || throw(ArgumentError("out_dims must be > 0, got $out_dims"))
    n_groups > 0 || throw(ArgumentError("n_groups must be > 0, got $n_groups"))
    return GroupedDense(activation, in_dims, out_dims, n_groups, init_weight, init_bias, use_bias)
end

function GroupedDense((in_dims, out_dims)::Pair{Int,Int}, n_groups::Int, activation=identity; kwargs...)
    return GroupedDense(in_dims, out_dims, n_groups, activation; kwargs...)
end

function Base.show(io::IO, l::GroupedDense)
    print(io, "GroupedDense($(l.in_dims) => $(l.out_dims), $(l.n_groups)")
    l.activation === identity || print(io, ", $(l.activation)")
    l.use_bias || print(io, ", use_bias=false")
    return print(io, ")")
end

function LuxCore.initialparameters(rng::AbstractRNG, l::GroupedDense)
    weight = l.init_weight(rng, l.out_dims, l.in_dims, l.n_groups)
    l.use_bias || return (; weight)
    return (; weight, bias=l.init_bias(rng, l.out_dims, l.n_groups, 1))
end

LuxCore.initialstates(::AbstractRNG, ::GroupedDense) = NamedTuple()

function LuxCore.parameterlength(l::GroupedDense)
    return l.out_dims * l.in_dims * l.n_groups + (l.use_bias ? l.out_dims * l.n_groups : 0)
end
LuxCore.statelength(::GroupedDense) = 0

function LuxCore.outputsize(l::GroupedDense, x, ::AbstractRNG)
    return (l.out_dims, size(x, 2))
end

function (l::GroupedDense)(x::AbstractArray{T,3}, ps, st::NamedTuple) where {T}
    # weight: [out, in, groups]; x: [in, groups, batch]
    # Contract over `in`; batch the group axis. Result is [out, batch, groups].
    y = batched_matmul(
        ps.weight,
        x;
        lhs_contracting_dim=2,
        rhs_contracting_dim=1,
        lhs_batching_dims=(3,),
        rhs_batching_dims=(2,),
    )
    y = permutedims(y, (1, 3, 2))
    if l.use_bias
        y = y .+ ps.bias
    end
    if l.activation !== identity
        y = l.activation.(y)
    end
    return y, st
end

end
