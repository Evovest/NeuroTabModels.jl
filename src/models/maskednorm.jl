module MaskedNorm

export MaskedBatchNorm, CarryMask, MaskSkip

using LuxCore
using Random: AbstractRNG

_is_training(st) = st.training === Val(true) || st.training === true

"""
    CarryMask(layer)

Pass a padding mask through a layer that only acts on the feature matrix:
`(x, mask) ↦ (layer(x), mask)`. Unmasked `x` is forwarded unchanged.
"""
struct CarryMask{L} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    layer::L
end
(l::CarryMask)(x::AbstractArray, ps, st) = l.layer(x, ps, st)
function (l::CarryMask)((x, mask)::Tuple, ps, st)
    y, st_ = l.layer(x, ps, st)
    return (y, mask), st_
end

"""
    MaskSkip(layer)

Skip-add that threads a padding mask: `(x, mask) ↦ (x + f(x), mask)`.
"""
struct MaskSkip{L} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    layer::L
end
function (l::MaskSkip)(x::AbstractMatrix, ps, st)
    y, st_ = l.layer(x, ps, st)
    return x .+ y, st_
end
function (l::MaskSkip)((x, mask)::Tuple, ps, st)
    y, st_ = l.layer((x, mask), ps, st)
    h = y isa Tuple ? y[1] : y
    return (x .+ h, mask), st_
end

"""
    MaskedBatchNorm(chs, act=identity; epsilon=1f-5, momentum=0.1f0)

BatchNorm over the last dimension of `(channels, tokens)`.

When the input is `(x, valid)` with `valid` a per-token flag, mean/variance and
running stats use only valid tokens so zero-padded group-buffer slots do not leak.
Unmasked `x` is ordinary BatchNorm.

Dense / Dropout in the same chain should be wrapped in [`CarryMask`](@ref);
residual skips should use [`MaskSkip`](@ref).
"""
struct MaskedBatchNorm{F} <: LuxCore.AbstractLuxLayer
    chs::Int
    act::F
    epsilon::Float32
    momentum::Float32
end
MaskedBatchNorm(chs::Int, act=identity; epsilon=1.0f-5, momentum=0.1f0) =
    MaskedBatchNorm{typeof(act)}(chs, act, Float32(epsilon), Float32(momentum))

function LuxCore.initialparameters(::AbstractRNG, l::MaskedBatchNorm)
    return (; scale=ones(Float32, l.chs), bias=zeros(Float32, l.chs))
end
function LuxCore.initialstates(::AbstractRNG, l::MaskedBatchNorm)
    return (; running_mean=zeros(Float32, l.chs), running_var=ones(Float32, l.chs), training=Val(true))
end

function _token_w(valid, x::AbstractMatrix)
    return reshape(eltype(x).(valid), 1, size(x, 2))
end

function _moments(x::AbstractMatrix, ::Nothing)
    T = eltype(x)
    μ = sum(x; dims=2) ./ T(size(x, 2))
    σ² = sum((x .- μ) .^ 2; dims=2) ./ T(size(x, 2))
    return μ, σ²
end
function _moments(x::AbstractMatrix, valid)
    T = eltype(x)
    w = _token_w(valid, x)
    nw = max(sum(w), one(T))
    μ = sum(x .* w; dims=2) ./ nw
    σ² = sum(((x .- μ) .^ 2) .* w; dims=2) ./ nw
    return μ, σ²
end

# 1 if running stats should take this batch, else 0. Arithmetic (no `if` on
# `any(valid)`) so Reactant can trace grouped / padded batches.
_running_gate(::Nothing, ::Type{T}) where {T} = one(T)
_running_gate(valid, ::Type{T}) where {T} = ifelse(any(valid), one(T), zero(T))

function _bn_apply(l::MaskedBatchNorm, x::AbstractMatrix, valid, ps, st)
    T = eltype(x)
    ϵ = T(l.epsilon)
    γ = reshape(ps.scale, :, 1)
    β = reshape(ps.bias, :, 1)
    if _is_training(st)
        μ, σ² = _moments(x, valid)
        y = l.act.(γ .* (x .- μ) ./ sqrt.(σ² .+ ϵ) .+ β)
        m = T(l.momentum)
        u = _running_gate(valid, T)
        rm = (1 - m * u) .* st.running_mean .+ (m * u) .* vec(μ)
        rv = (1 - m * u) .* st.running_var .+ (m * u) .* vec(σ²)
        return y, (; running_mean=rm, running_var=rv, training=st.training)
    end
    μ = reshape(st.running_mean, :, 1)
    σ² = reshape(st.running_var, :, 1)
    y = l.act.(γ .* (x .- μ) ./ sqrt.(σ² .+ ϵ) .+ β)
    return y, st
end

(l::MaskedBatchNorm)(x::AbstractMatrix, ps, st) = _bn_apply(l, x, nothing, ps, st)
function (l::MaskedBatchNorm)((x, valid)::Tuple, ps, st)
    y, st_ = _bn_apply(l, x, valid, ps, st)
    return (y, valid), st_
end

end
