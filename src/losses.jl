module Losses

export get_loss_fn, get_loss_type
export LossType, MSE, MAE, LogLoss, MLogLoss, GaussianMLE, Tweedie
export reduce_pred

import Statistics: mean
import NNlib: logsigmoid, logsoftmax

abstract type LossType end
abstract type MSE <: LossType end
abstract type MAE <: LossType end
abstract type LogLoss <: LossType end
abstract type MLogLoss <: LossType end
abstract type GaussianMLE <: LossType end
abstract type Tweedie <: LossType end

"""
    reduce_pred(y)
For 3D ensemble output `(D, K, B)`, averages over K → `(D, B)`.
"""
reduce_pred(y::AbstractArray{T,3}) where {T} =
    dropdims(mean(y; dims=2); dims=2)

_bcast(y::AbstractVector) = reshape(y, 1, 1, :)
_bcast(y::AbstractMatrix) = reshape(y, size(y, 1), 1, size(y, 2))

function mse_loss(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    loss = mean((pred .- _bcast(data[2])) .^ 2)
    return loss, st_, NamedTuple()
end
function mse_loss(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    r2 = mean((pred .- _bcast(data[2])) .^ 2; dims=2)
    w = _bcast(data[3])
    return sum(r2 .* w) / sum(w), st_, NamedTuple()
end
function mse_loss(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    r2 = mean((pred .+ _bcast(data[4]) .- _bcast(data[2])) .^ 2; dims=2)
    w = _bcast(data[3])
    return sum(r2 .* w) / sum(w), st_, NamedTuple()
end

function mae_loss(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    loss = mean(abs.(pred .- _bcast(data[2])))
    return loss, st_, NamedTuple()
end
function mae_loss(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    ae = mean(abs.(pred .- _bcast(data[2])); dims=2)
    w = _bcast(data[3])
    return sum(ae .* w) / sum(w), st_, NamedTuple()
end
function mae_loss(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    ae = mean(abs.(pred .+ _bcast(data[4]) .- _bcast(data[2])); dims=2)
    w = _bcast(data[3])
    return sum(ae .* w) / sum(w), st_, NamedTuple()
end

function logloss(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    y = _bcast(data[2])
    loss = mean((1 .- y) .* pred .- logsigmoid.(pred))
    return loss, st_, NamedTuple()
end
function logloss(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    y = _bcast(data[2]); w = _bcast(data[3])
    per_head = mean((1 .- y) .* pred .- logsigmoid.(pred); dims=2)
    return sum(per_head .* w) / sum(w), st_, NamedTuple()
end
function logloss(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    p = pred .+ _bcast(data[4])
    y = _bcast(data[2]); w = _bcast(data[3])
    per_head = mean((1 .- y) .* p .- logsigmoid.(p); dims=2)
    return sum(per_head .* w) / sum(w), st_, NamedTuple()
end

function mlogloss(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    k = size(pred, 1)
    y_oh = (UInt32(1):UInt32(k)) .== reshape(data[2], 1, 1, :)
    lsm = logsoftmax(pred; dims=1)
    loss = mean(-sum(y_oh .* lsm; dims=1))
    return loss, st_, NamedTuple()
end
function mlogloss(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    k = size(pred, 1)
    y_oh = (UInt32(1):UInt32(k)) .== reshape(data[2], 1, 1, :)
    lsm = logsoftmax(pred; dims=1)
    per_head = mean(-sum(y_oh .* lsm; dims=1); dims=2)
    w = _bcast(data[3])
    return sum(per_head .* w) / sum(w), st_, NamedTuple()
end
function mlogloss(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred = pred .+ _bcast(data[4])
    k = size(pred, 1)
    y_oh = (UInt32(1):UInt32(k)) .== reshape(data[2], 1, 1, :)
    lsm = logsoftmax(pred; dims=1)
    per_head = mean(-sum(y_oh .* lsm; dims=1); dims=2)
    w = _bcast(data[3])
    return sum(per_head .* w) / sum(w), st_, NamedTuple()
end

function tweedie(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    rho = eltype(pred)(1.5)
    ep = exp.(pred); y = _bcast(data[2])
    loss = mean(2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* ep .^ (1 - rho) / (1 - rho) .+
               ep .^ (2 - rho) / (2 - rho)))
    return loss, st_, NamedTuple()
end
function tweedie(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    rho = eltype(pred)(1.5)
    ep = exp.(pred); y = _bcast(data[2]); w = _bcast(data[3])
    dev = 2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* ep .^ (1 - rho) / (1 - rho) .+
               ep .^ (2 - rho) / (2 - rho))
    per_head = mean(dev; dims=2)
    return sum(per_head .* w) / sum(w), st_, NamedTuple()
end
function tweedie(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    rho = eltype(pred)(1.5)
    ep = exp.(pred .+ _bcast(data[4])); y = _bcast(data[2]); w = _bcast(data[3])
    dev = 2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* ep .^ (1 - rho) / (1 - rho) .+
               ep .^ (2 - rho) / (2 - rho))
    per_head = mean(dev; dims=2)
    return sum(per_head .* w) / sum(w), st_, NamedTuple()
end

gaussian_mle_loss(μ, σ, y) =
    mean(σ .+ (y .- μ) .^ 2 ./ (2 .* max.(oftype.(σ, 2e-7), exp.(2 .* σ))))

gaussian_mle_loss(μ, σ, y, w) =
    sum((σ .+ (y .- μ) .^ 2 ./ (2 .* max.(oftype.(σ, 2e-7), exp.(2 .* σ)))) .* w) / sum(w)

function gaussian_mle(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    μ = pred[1:1, :, :]; σ = pred[2:2, :, :]
    loss = gaussian_mle_loss(μ, σ, _bcast(data[2]))
    return loss, st_, NamedTuple()
end
function gaussian_mle(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    μ = pred[1:1, :, :]; σ = pred[2:2, :, :]
    loss = gaussian_mle_loss(μ, σ, _bcast(data[2]), _bcast(data[3]))
    return loss, st_, NamedTuple()
end
function gaussian_mle(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred = pred .+ _bcast(data[4])
    μ = pred[1:1, :, :]; σ = pred[2:2, :, :]
    loss = gaussian_mle_loss(μ, σ, _bcast(data[2]), _bcast(data[3]))
    return loss, st_, NamedTuple()
end

get_loss_fn(::Type{<:MSE}) = mse_loss
get_loss_fn(::Type{<:MAE}) = mae_loss
get_loss_fn(::Type{<:LogLoss}) = logloss
get_loss_fn(::Type{<:MLogLoss}) = mlogloss
get_loss_fn(::Type{<:GaussianMLE}) = gaussian_mle
get_loss_fn(::Type{<:Tweedie}) = tweedie

const _loss_type_dict = Dict(
    :mse => MSE,
    :mae => MAE,
    :logloss => LogLoss,
    :tweedie => Tweedie,
    :gaussian_mle => GaussianMLE,
    :mlogloss => MLogLoss,
)

get_loss_type(loss::Symbol) = _loss_type_dict[loss]
get_loss_fn(s::Symbol) = get_loss_fn(get_loss_type(s))

end