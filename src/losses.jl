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

function mse_loss(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    p = vec(pred); y = vec(data[2])
    return mean((p .- y) .^ 2), st_, NamedTuple()
end
function mse_loss(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    p = vec(pred); y = vec(data[2]); w = vec(data[3])
    return sum((p .- y) .^ 2 .* w) / sum(w), st_, NamedTuple()
end
function mse_loss(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    p = vec(pred) .+ vec(data[4]); y = vec(data[2]); w = vec(data[3])
    return sum((p .- y) .^ 2 .* w) / sum(w), st_, NamedTuple()
end

function mae_loss(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    p = vec(pred); y = vec(data[2])
    return mean(abs.(p .- y)), st_, NamedTuple()
end
function mae_loss(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    p = vec(pred); y = vec(data[2]); w = vec(data[3])
    return sum(abs.(p .- y) .* w) / sum(w), st_, NamedTuple()
end
function mae_loss(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    p = vec(pred) .+ vec(data[4]); y = vec(data[2]); w = vec(data[3])
    return sum(abs.(p .- y) .* w) / sum(w), st_, NamedTuple()
end

function logloss(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    p = vec(pred); y = vec(data[2])
    return mean((1 .- y) .* p .- logsigmoid.(p)), st_, NamedTuple()
end
function logloss(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    p = vec(pred); y = vec(data[2]); w = vec(data[3])
    return sum(w .* ((1 .- y) .* p .- logsigmoid.(p))) / sum(w), st_, NamedTuple()
end
function logloss(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    p = vec(pred) .+ vec(data[4]); y = vec(data[2]); w = vec(data[3])
    return sum(w .* ((1 .- y) .* p .- logsigmoid.(p))) / sum(w), st_, NamedTuple()
end

function mlogloss(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    k = size(pred, 1)
    y_oh = (UInt32(1):UInt32(k)) .== reshape(data[2], 1, :)
    lsm = logsoftmax(pred; dims=1)
    return mean(-sum(y_oh .* lsm; dims=1)), st_, NamedTuple()
end
function mlogloss(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    k = size(pred, 1)
    y_oh = (UInt32(1):UInt32(k)) .== reshape(data[2], 1, :)
    lsm = logsoftmax(pred; dims=1)
    return sum(vec(-sum(y_oh .* lsm; dims=1)) .* vec(data[3])) / sum(data[3]), st_, NamedTuple()
end
function mlogloss(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred = pred .+ data[4]
    k = size(pred, 1)
    y_oh = (UInt32(1):UInt32(k)) .== reshape(data[2], 1, :)
    lsm = logsoftmax(pred; dims=1)
    return sum(vec(-sum(y_oh .* lsm; dims=1)) .* vec(data[3])) / sum(data[3]), st_, NamedTuple()
end

function tweedie(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    rho = eltype(data[1])(1.5)
    ep = exp.(vec(pred)); y = vec(data[2])
    loss = mean(2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* ep .^ (1 - rho) / (1 - rho) .+
               ep .^ (2 - rho) / (2 - rho)))
    return loss, st_, NamedTuple()
end
function tweedie(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    rho = eltype(data[1])(1.5)
    ep = exp.(vec(pred)); y = vec(data[2]); w = vec(data[3])
    loss = sum(w .* 2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* ep .^ (1 - rho) / (1 - rho) .+
                   ep .^ (2 - rho) / (2 - rho))) / sum(w)
    return loss, st_, NamedTuple()
end
function tweedie(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    rho = eltype(data[1])(1.5)
    ep = exp.(vec(pred) .+ vec(data[4])); y = vec(data[2]); w = vec(data[3])
    loss = sum(w .* 2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* ep .^ (1 - rho) / (1 - rho) .+
                   ep .^ (2 - rho) / (2 - rho))) / sum(w)
    return loss, st_, NamedTuple()
end

gaussian_mle_loss(μ::AbstractVector, σ::AbstractVector, y::AbstractVector) =
    -sum(-σ .- (y .- μ) .^ 2 ./ (2 .* max.(oftype.(σ, 2e-7), exp.(2 .* σ))))

gaussian_mle_loss(μ::AbstractVector, σ::AbstractVector, y::AbstractVector, w::AbstractVector) =
    -sum((-σ .- (y .- μ) .^ 2 ./ (2 .* max.(oftype.(σ, 2e-7), exp.(2 .* σ)))) .* w) / sum(w)

function gaussian_mle(model, ps, st, data::Tuple{Any,Any})
    pred, st_ = model(data[1], ps, st)
    loss = gaussian_mle_loss(view(pred, 1, :), view(pred, 2, :), vec(data[2]))
    return loss, st_, NamedTuple()
end
function gaussian_mle(model, ps, st, data::Tuple{Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    loss = gaussian_mle_loss(view(pred, 1, :), view(pred, 2, :), vec(data[2]), vec(data[3]))
    return loss, st_, NamedTuple()
end
function gaussian_mle(model, ps, st, data::Tuple{Any,Any,Any,Any})
    pred, st_ = model(data[1], ps, st)
    pred = pred .+ data[4]
    loss = gaussian_mle_loss(view(pred, 1, :), view(pred, 2, :), vec(data[2]), vec(data[3]))
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