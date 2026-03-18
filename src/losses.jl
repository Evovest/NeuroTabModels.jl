module Losses

export get_loss_fn, get_loss_type
export LossType, MSE, MAE, LogLoss, MLogLoss, GaussianMLE, Tweedie

import Statistics: mean
import NNlib: logsigmoid, logsoftmax

abstract type LossType end
abstract type MSE <: LossType end
abstract type MAE <: LossType end
abstract type LogLoss <: LossType end
abstract type MLogLoss <: LossType end
abstract type GaussianMLE <: LossType end
abstract type Tweedie <: LossType end

_reshape_3d(x::AbstractVector) = reshape(x, 1, 1, :)
_reshape_3d(x::AbstractMatrix) = reshape(x, size(x, 1), 1, size(x, 2))
_reshape_3d(x::AbstractArray{T,3}) where {T} = x

function _forward(model, ps, st, x)
    pred, st_ = model(x, ps, st)
    return _reshape_3d(pred), st_
end

_reduce(loss) = mean(loss)
_reduce(loss, w) = sum(mean(loss; dims=2) .* w) / sum(w)

function _apply_loss(core, model, ps, st, data::Tuple{Any,Any})
    pred, st_ = _forward(model, ps, st, data[1])
    return _reduce(core(pred, _reshape_3d(data[2]))), st_, NamedTuple()
end
function _apply_loss(core, model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = _forward(model, ps, st, data[1])
    return _reduce(core(pred, _reshape_3d(data[2])), _reshape_3d(data[3])), st_, NamedTuple()
end
function _apply_loss(core, model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = _forward(model, ps, st, data[1])
    return _reduce(core(pred .+ _reshape_3d(data[4]), _reshape_3d(data[2])), _reshape_3d(data[3])), st_, NamedTuple()
end

_mse_core(pred, y) = (pred .- y) .^ 2
_mae_core(pred, y) = abs.(pred .- y)
_logloss_core(pred, y) = (1 .- y) .* pred .- logsigmoid.(pred)

function _mlogloss_core(pred, y)
    nclasses = size(pred, 1)
    classes = reshape(Int32(1):Int32(nclasses), :, 1, 1)
    y_idx = reshape(Int32.(y), 1, 1, :)
    y_oh = Float32.(classes .== y_idx)
    return -sum(y_oh .* logsoftmax(pred; dims=1); dims=1)
end

function _tweedie_core(pred, y)
    rho = eltype(pred)(1.5)
    ep = exp.(pred)
    2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* ep .^ (1 - rho) / (1 - rho) .+
          ep .^ (2 - rho) / (2 - rho))
end

mse_loss(m, ps, st, d) = _apply_loss(_mse_core, m, ps, st, d)
mae_loss(m, ps, st, d) = _apply_loss(_mae_core, m, ps, st, d)
logloss(m, ps, st, d) = _apply_loss(_logloss_core, m, ps, st, d)
mlogloss(m, ps, st, d) = _apply_loss(_mlogloss_core, m, ps, st, d)
tweedie(m, ps, st, d) = _apply_loss(_tweedie_core, m, ps, st, d)

function _gaussian_mle_core(μ, σ, y)
    σ .+ (y .- μ) .^ 2 ./ (2 .* max.(eltype(σ)(2e-7), exp.(2 .* σ)))
end

function gaussian_mle(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = _forward(model, ps, st, data[1])
    return _reduce(_gaussian_mle_core(pred[1:1,:,:], pred[2:2,:,:], _reshape_3d(data[2]))), st_, NamedTuple()
end
function gaussian_mle(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = _forward(model, ps, st, data[1])
    return _reduce(_gaussian_mle_core(pred[1:1,:,:], pred[2:2,:,:], _reshape_3d(data[2])), _reshape_3d(data[3])), st_, NamedTuple()
end
function gaussian_mle(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = _forward(model, ps, st, data[1])
    pred = pred .+ _reshape_3d(data[4])
    return _reduce(_gaussian_mle_core(pred[1:1,:,:], pred[2:2,:,:], _reshape_3d(data[2])), _reshape_3d(data[3])), st_, NamedTuple()
end

get_loss_fn(::Type{<:MSE}) = mse_loss
get_loss_fn(::Type{<:MAE}) = mae_loss
get_loss_fn(::Type{<:LogLoss}) = logloss
get_loss_fn(::Type{<:MLogLoss}) = mlogloss
get_loss_fn(::Type{<:GaussianMLE}) = gaussian_mle
get_loss_fn(::Type{<:Tweedie}) = tweedie

const _loss_type_dict = Dict(
    :mse => MSE, :mae => MAE, :logloss => LogLoss,
    :mlogloss => MLogLoss, :gaussian_mle => GaussianMLE, :tweedie => Tweedie,
)

get_loss_type(loss::Symbol) = _loss_type_dict[loss]
get_loss_fn(s::Symbol) = get_loss_fn(get_loss_type(s))

end