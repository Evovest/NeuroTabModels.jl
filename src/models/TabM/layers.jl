"""
    EnsembleView(k)

Repeat `(D, B)` input to `(D, K, B)`. Passes through `(D, K, B)` unchanged.
"""
struct EnsembleView <: LuxCore.AbstractLuxLayer
    k::Int
end

function (m::EnsembleView)(x::AbstractMatrix, ps, st)
    D, B = size(x)
    return repeat(reshape(x, D, 1, B), 1, m.k, 1), st
end

function (m::EnsembleView)(x::AbstractArray{T,3}, ps, st) where {T}
    # @assert size(x, 2) == m.k "Expected K=$(m.k), got $(size(x, 2))"
    return x, st
end

"""
    LinearBatchEnsemble(in_f, out_f; k, scaling_init=:random_signs, bias=true)

Batch-ensemble linear: `y = S ⊙ (W(R ⊙ x)) + bias`.

# Arguments
- `in_f`, `out_f`: Input and output dimensions.
- `k::Int`: Ensemble size.
- `scaling_init`: `:ones`, `:normal`, or `:random_signs`; or `(R, S)` tuple.
- `bias::Bool`: Per-member bias (default `true`).
"""
struct LinearBatchEnsemble <: LuxCore.AbstractLuxLayer
    in_features::Int
    out_features::Int
    k::Int
    use_bias::Bool
    r_init::Symbol
    s_init::Symbol
end

function LinearBatchEnsemble(in_f::Int, out_f::Int;
    k::Int,
    scaling_init::Union{Symbol,Tuple{Symbol,Symbol}}=:ones,
    bias::Bool=true)
    r_init, s_init = scaling_init isa Tuple ? scaling_init : (scaling_init, scaling_init)
    return LinearBatchEnsemble(in_f, out_f, k, bias, r_init, s_init)
end

function LuxCore.initialparameters(rng::AbstractRNG, m::LinearBatchEnsemble)
    weight = _init_rsqrt_uniform(rng, (m.out_features, m.in_features), m.in_features)

    r = _init_scaling(rng, (m.in_features, m.k), m.r_init)
    s = _init_scaling(rng, (m.out_features, m.k), m.s_init)

    d = (; weight, r, s)
    if m.use_bias
        bias = _init_rsqrt_uniform(rng, (m.out_features, m.k), m.in_features)
        d = merge(d, (; bias))
    end
    return d
end

function (m::LinearBatchEnsemble)(x::AbstractArray{T,3}, ps, st) where {T}
    in_f, k, batch = size(x)
    x = x .* reshape(ps.r, m.in_features, m.k, 1)
    x = reshape(ps.weight * reshape(x, in_f, k * batch), m.out_features, k, batch)
    x = x .* reshape(ps.s, m.out_features, m.k, 1)
    if m.use_bias
        x = x .+ reshape(ps.bias, m.out_features, m.k, 1)
    end
    return x, st
end

"""
    LinearEnsemble(in_f, out_f, k; bias=true)

`k` independent linear layers via `batched_matmul`. Input/output `(features, k, batch)`.
"""
struct LinearEnsemble <: LuxCore.AbstractLuxLayer
    in_features::Int
    out_features::Int
    k::Int
    use_bias::Bool
end

LinearEnsemble(in_f::Int, out_f::Int, k::Int; bias::Bool=true) =
    LinearEnsemble(in_f, out_f, k, bias)

function LuxCore.initialparameters(rng::AbstractRNG, m::LinearEnsemble)
    d = (; weight=_init_rsqrt_uniform(rng, (m.out_features, m.in_features, m.k), m.in_features))
    if m.use_bias
        d = merge(d, (; bias=_init_rsqrt_uniform(rng, (m.out_features, m.k), m.in_features)))
    end
    return d
end

function (m::LinearEnsemble)(x::AbstractArray{T,3}, ps, st) where {T}
    xp = permutedims(x, (1, 3, 2))
    out = batched_matmul(ps.weight, xp)
    out = permutedims(out, (1, 3, 2))
    if m.use_bias
        out = out .+ reshape(ps.bias, m.out_features, m.k, 1)
    end
    return out, st
end

"""
    ScaleEnsemble(k, d; init=:random_signs, bias=false)

Per-member elementwise scaling on `(d, k, batch)` input.
"""
struct ScaleEnsemble <: LuxCore.AbstractLuxLayer
    k::Int
    d::Int
    init::Symbol
    use_bias::Bool
end

function ScaleEnsemble(k::Int, d::Int;
    init::Symbol=:random_signs,
    bias::Bool=false)
    return ScaleEnsemble(k, d, init, bias)
end

function LuxCore.initialparameters(rng::AbstractRNG, m::ScaleEnsemble)
    weight = _init_scaling(rng, (m.d, m.k), m.init)
    d = (; weight)
    if m.use_bias
        d = merge(d, (; bias=zeros(Float32, m.d, m.k)))
    end
    return d
end

function (m::ScaleEnsemble)(x::AbstractArray{T,3}, ps, st) where {T}
    w = reshape(ps.weight, m.d, m.k, 1)
    if m.use_bias
        return reshape(ps.bias, m.d, m.k, 1) .+ w .* x, st
    else
        return x .* w, st
    end
end

function _init_rsqrt_uniform(rng::AbstractRNG, dims, d::Int)
    s = Float32(1 / sqrt(d))
    return s .* (2f0 .* rand(rng, Float32, dims...) .- 1f0)
end

function _init_scaling(rng::AbstractRNG, dims, init::Symbol)
    if init == :ones
        return ones(Float32, dims...)
    elseif init == :normal
        return randn(rng, Float32, dims...)
    elseif init == :random_signs
        return Float32.(2 .* (rand(rng, Float32, dims...) .> 0.5f0) .- 1)
    else
        error("Unknown scaling init: $init")
    end
end

# function _init_scaling_with_chunks(rng::AbstractRNG, dims::Tuple{Int,Int},
#     init::Symbol, chunks::Vector{Int})
#     d, k = dims
#     @assert d == sum(chunks) "Chunks must sum to $d, got $(sum(chunks))"
#     weight = zeros(Float32, d, k)
#     row = 1
#     for chunk_size in chunks
#         val = _init_scaling(rng, (1, k), init)
#         weight[row:(row+chunk_size-1), :] .= repeat(val, chunk_size, 1)
#         row += chunk_size
#     end
#     return weight
# end
