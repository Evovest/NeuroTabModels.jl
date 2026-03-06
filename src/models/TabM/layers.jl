using Lux: AbstractLuxLayer, WrappedFunction
using LuxCore
using LuxLib: batched_matmul
using Random: AbstractRNG
using NNlib: relu

"""
    _init_rsqrt_uniform(rng, dims, d) → Array{Float32}

Uniform init in `[-1/√d, 1/√d]`.
"""
function _init_rsqrt_uniform(rng::AbstractRNG, dims, d::Int)
    s = Float32(1 / sqrt(d))
    return s .* (2f0 .* rand(rng, Float32, dims...) .- 1f0)
end

"""
    _init_scaling(rng, dims, init) → Array{Float32}

Initialize scaling vectors for ensemble adapters.
- `:ones` — deterministic ones
- `:normal` — N(0,1)
- `:random_signs` — ±1
"""
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

"""
    _init_scaling_with_chunks(rng, dims, init, chunks) → Array{Float32}

Initialize scaling with grouped chunks. Each chunk shares the same random value
per ensemble member, providing structured diversity for features with different
representation sizes.
"""
function _init_scaling_with_chunks(rng::AbstractRNG, dims::Tuple{Int,Int},
                                    init::Symbol, chunks::Vector{Int})
    d, k = dims
    @assert d == sum(chunks) "Chunks must sum to $d, got $(sum(chunks))"
    weight = zeros(Float32, d, k)
    row = 1
    for chunk_size in chunks
        val = _init_scaling(rng, (1, k), init)
        weight[row:row+chunk_size-1, :] .= repeat(val, chunk_size, 1)
        row += chunk_size
    end
    return weight
end

_broadcast_relu(x) = relu.(x)

"""
    EnsembleView(k)

Broadcasts a `(D, B)` matrix to `(D, K, B)` by repeating along dim 2.
Passes through `(D, K, B)` input unchanged.
"""
struct EnsembleView <: AbstractLuxLayer
    k::Int
end

function (m::EnsembleView)(x::AbstractMatrix, ps, st)
    D, B = size(x)
    return repeat(reshape(x, D, 1, B), 1, m.k, 1), st
end

function (m::EnsembleView)(x::AbstractArray{T,3}, ps, st) where {T}
    @assert size(x, 2) == m.k "Expected K=$(m.k), got $(size(x, 2))"
    return x, st
end

"""
    LinearBatchEnsemble(in_f, out_f; k, scaling_init=:random_signs,
                        first_scaling_init_chunks=nothing, bias=true)

Batch-ensemble linear: `y = S ⊙ (W(R ⊙ x)) + bias` with shared `W` and
per-member `R`, `S`, `bias`.

Equivalent to defining per-member weight matrices `Wᵢ = W ⊙ (sᵢrᵢᵀ)`.

# Arguments
- `in_f::Int`: Input dimension.
- `out_f::Int`: Output dimension.
- `k::Int`: Number of ensemble members.
- `scaling_init`: Init for R/S. A `Symbol` (applied to both) or
  `Tuple{Symbol,Symbol}` for `(R, S)`. Options: `:ones`, `:normal`, `:random_signs`.
- `first_scaling_init_chunks`: Chunk sizes for grouped R init.
- `bias::Bool`: Include per-member bias (default `true`).
"""
struct LinearBatchEnsemble <: AbstractLuxLayer
    in_features::Int
    out_features::Int
    k::Int
    use_bias::Bool
    r_init::Symbol
    s_init::Symbol
    r_init_chunks::Union{Nothing, Vector{Int}}
end

function LinearBatchEnsemble(in_f::Int, out_f::Int;
        k::Int,
        scaling_init::Union{Symbol, Tuple{Symbol,Symbol}} = :random_signs,
        first_scaling_init_chunks::Union{Nothing, Vector{Int}} = nothing,
        bias::Bool = true)
    r_init, s_init = scaling_init isa Tuple ? scaling_init : (scaling_init, scaling_init)
    return LinearBatchEnsemble(in_f, out_f, k, bias, r_init, s_init, first_scaling_init_chunks)
end

function LuxCore.initialparameters(rng::AbstractRNG, m::LinearBatchEnsemble)
    weight = _init_rsqrt_uniform(rng, (m.out_features, m.in_features), m.in_features)

    r = if m.r_init_chunks !== nothing
        _init_scaling_with_chunks(rng, (m.in_features, m.k), m.r_init, m.r_init_chunks)
    else
        _init_scaling(rng, (m.in_features, m.k), m.r_init)
    end
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
    SharedDense(in_features, out_features)

Standard linear layer that broadcasts over the K (ensemble) dimension.
Shared weight and bias across all members.

# Arguments
- `in_features::Int`: Input dimension.
- `out_features::Int`: Output dimension.
"""
struct SharedDense <: AbstractLuxLayer
    in_features::Int
    out_features::Int
end

function LuxCore.initialparameters(rng::AbstractRNG, m::SharedDense)
    return (;
        weight = _init_rsqrt_uniform(rng, (m.out_features, m.in_features), m.in_features),
        bias = _init_rsqrt_uniform(rng, (m.out_features,), m.in_features),
    )
end

function (m::SharedDense)(x::AbstractArray{T,3}, ps, st) where {T}
    d_in, k, batch = size(x)
    out = ps.weight * reshape(x, d_in, k * batch) .+ ps.bias
    return reshape(out, m.out_features, k, batch), st
end

"""
    LinearEnsemble(in_f, out_f, k; bias=true)

`k` independent linear layers applied via `batched_matmul`.

# Arguments
- `in_f::Int`: Input dimension.
- `out_f::Int`: Output dimension.
- `k::Int`: Number of ensemble members.
- `bias::Bool`: Include per-member bias (default `true`).
"""
struct LinearEnsemble <: AbstractLuxLayer
    in_features::Int
    out_features::Int
    k::Int
    use_bias::Bool
end

LinearEnsemble(in_f::Int, out_f::Int, k::Int; bias::Bool = true) =
    LinearEnsemble(in_f, out_f, k, bias)

function LuxCore.initialparameters(rng::AbstractRNG, m::LinearEnsemble)
    d = (; weight = _init_rsqrt_uniform(rng, (m.out_features, m.in_features, m.k), m.in_features))
    if m.use_bias
        d = merge(d, (; bias = _init_rsqrt_uniform(rng, (m.out_features, m.k), m.in_features)))
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
    ScaleEnsemble(k, d; init=:random_signs, init_chunks=nothing, bias=false)

Per-member elementwise scaling (and optional bias).

# Arguments
- `k::Int`: Number of ensemble members.
- `d::Int`: Feature dimension.
- `init::Symbol`: Weight init — `:ones`, `:normal`, or `:random_signs`.
- `init_chunks`: Chunk sizes for grouped init.
- `bias::Bool`: Include per-member additive bias (default `false`).
"""
struct ScaleEnsemble <: AbstractLuxLayer
    k::Int
    d::Int
    init::Symbol
    init_chunks::Union{Nothing, Vector{Int}}
    use_bias::Bool
end

function ScaleEnsemble(k::Int, d::Int;
        init::Symbol = :random_signs,
        init_chunks::Union{Nothing, Vector{Int}} = nothing,
        bias::Bool = false)
    return ScaleEnsemble(k, d, init, init_chunks, bias)
end

function LuxCore.initialparameters(rng::AbstractRNG, m::ScaleEnsemble)
    weight = if m.init_chunks !== nothing
        _init_scaling_with_chunks(rng, (m.d, m.k), m.init, m.init_chunks)
    else
        _init_scaling(rng, (m.d, m.k), m.init)
    end
    d = (; weight)
    if m.use_bias
        d = merge(d, (; bias = zeros(Float32, m.d, m.k)))
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

"""
    MeanEnsemble()

Averages over the ensemble (K) dimension: `(D, K, B)` → `(D, B)`.
Equivalent to `reduce_pred` but as a Lux layer.
"""
struct MeanEnsemble <: AbstractLuxLayer end

function (::MeanEnsemble)(x::AbstractArray{T,3}, ps, st) where {T}
    k = size(x, 2)
    return dropdims(sum(x; dims=2); dims=2) ./ T(k), st
end