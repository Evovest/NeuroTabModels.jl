module Losses

export LossType, MSE, MAE, LogLoss, MLogLoss, GaussianMLE, Tweedie, Correlation
export masked_input

import Statistics: mean
import NNlib: logsigmoid, logsoftmax

"""
    LossType

Abstract supertype for training losses. Concrete subtypes are callable functors
compatible with Lux's training API:

```julia
loss(model, ps, st, data) -> (scalar, updated_state, NamedTuple())
```

`data` is `(x, y)`, `(x, y, w)`, or `(x, y, w, offset)`. Construct from a symbol
with `LossType(:mse)` (returns `MSE()`), or use the structs directly (`MSE()`).
"""
abstract type LossType <: Function end

struct MSE <: LossType end
struct MAE <: LossType end
struct LogLoss <: LossType end
struct MLogLoss <: LossType end
struct GaussianMLE <: LossType end
struct Tweedie <: LossType end
struct Correlation <: LossType end

LossType(loss::LossType) = loss
LossType(::Type{L}) where {L<:LossType} = L()
LossType(s::Symbol) = LossType(Val(s))
LossType(::Val{:mse}) = MSE()
LossType(::Val{:mae}) = MAE()
LossType(::Val{:logloss}) = LogLoss()
LossType(::Val{:mlogloss}) = MLogLoss()
LossType(::Val{:gaussian_mle}) = GaussianMLE()
LossType(::Val{:tweedie}) = Tweedie()
LossType(::Val{:correlation}) = Correlation()
function LossType(::Val{s}) where {s}
    error(
        "Unknown loss `:$s`. Supported: :mse, :mae, :logloss, :mlogloss, :gaussian_mle, :tweedie, :correlation.",
    )
end

"""Number of model output channels. `MLogLoss` is the class count and is set in `fit`."""
noutputs(::LossType) = 1
noutputs(::GaussianMLE) = 2

"""Whether this loss standardises the target when `scale_target=true`."""
scales_target(::LossType) = false
scales_target(::Union{MSE,MAE,GaussianMLE,Correlation}) = true

_reshape_3d(x::AbstractVector) = reshape(x, 1, 1, :)
_reshape_3d(x::AbstractMatrix) = reshape(x, size(x, 1), 1, size(x, 2))
_reshape_3d(x::AbstractArray{T,3}) where {T} = x

function _forward(model, ps, st, x)
    pred, st_ = model(x, ps, st)
    return _reshape_3d(pred), st_
end

"""
    masked_input(model, x, w)

Features to pass into `model` when a sample-weight / padding mask `w` is available.
Default ignores `w` (same as calling `model(x)`). Architectures that mix across the
batch override to return `(x, w)`.
"""
masked_input(::Any, x, w) = x

_model_x(model, data::Tuple{Any,Any}) = data[1]
_model_x(model, data::Tuple{Any,Any,Any}) = masked_input(model, data[1], data[3])
_model_x(model, data::Tuple{Any,Any,Any,Any}) = masked_input(model, data[1], data[3])

_target(data) = data[2]
_weight(data::Tuple{Any,Any}) = nothing
_weight(data::Tuple{Any,Any,Any}) = data[3]
_weight(data::Tuple{Any,Any,Any,Any}) = data[3]
_offset(data::Tuple{Any,Any}) = nothing
_offset(data::Tuple{Any,Any,Any}) = nothing
_offset(data::Tuple{Any,Any,Any,Any}) = data[4]

_apply_offset(pred, ::Nothing) = pred
_apply_offset(pred, offset) = pred .+ _reshape_3d(offset)

_reduce(loss) = mean(loss)
_reduce(loss, w) = sum(mean(loss; dims=2) .* w) / sum(w)

_aggregate(loss, pred, y, ::Nothing) = _reduce(_pointwise(loss, pred, y))
_aggregate(loss, pred, y, w) = _reduce(_pointwise(loss, pred, y), _reshape_3d(w))

function (loss::LossType)(model, ps, st, data)
    pred, st_ = _forward(model, ps, st, _model_x(model, data))
    pred = _apply_offset(pred, _offset(data))
    return _aggregate(loss, pred, _reshape_3d(_target(data)), _weight(data)), st_, NamedTuple()
end

_pointwise(::MSE, pred, y) = (pred .- y) .^ 2
_pointwise(::MAE, pred, y) = abs.(pred .- y)
_pointwise(::LogLoss, pred, y) = (1 .- y) .* pred .- logsigmoid.(pred)

function _pointwise(::MLogLoss, pred, y)
    nclasses = size(pred, 1)
    classes = reshape(Int32(1):Int32(nclasses), :, 1, 1)
    y_idx = reshape(Int32.(y), 1, 1, :)
    lsm = logsoftmax(pred; dims=1)
    return -sum(ifelse.(classes .== y_idx, lsm, zero(eltype(lsm))); dims=1)
end

function _pointwise(::Tweedie, pred, y)
    rho = eltype(pred)(1.5)
    ep = exp.(pred)
    2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* ep .^ (1 - rho) / (1 - rho) .+ ep .^ (2 - rho) / (2 - rho))
end

function _pointwise(::GaussianMLE, pred, y)
    μ = pred[1:1, :, :]
    σ = pred[2:2, :, :]
    σ .+ (y .- μ) .^ 2 ./ (2 .* max.(eltype(σ)(2e-7), exp.(2 .* σ)))
end

# First output channel (`μ` when the head is 2-wide), mean over the ensemble axis.
_corr_from_pred(pred) = vec(mean(view(pred, 1, :, :); dims=1))
_ones_like(p) = fill!(similar(p), one(eltype(p)))

function _correlation_value(p, y, w)
    p = vec(p)
    y = vec(y)
    w = vec(w)
    sw = sum(w)
    p_mean = (w' * p) / sw
    p_var = (w' * (p .^ 2)) / sw - p_mean^2
    y_mean = (w' * y) / sw
    y_var = (w' * (y .^ 2)) / sw - y_mean^2
    py_mean = (w' * (p .* y)) / sw
    return (py_mean - p_mean * y_mean) / (sqrt(p_var) * sqrt(y_var)) * sw
end

function _aggregate(::Correlation, pred, y, ::Nothing)
    p = _corr_from_pred(pred)
    return -_correlation_value(p, y, _ones_like(p))
end
function _aggregate(::Correlation, pred, y, w)
    p = _corr_from_pred(pred)
    return -_correlation_value(p, y, w)
end

end
