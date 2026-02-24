module Losses

export get_loss_fn, get_loss_type
export LossType, MSE, MAE, LogLoss, MLogLoss, GaussianMLE, Tweedie

import Statistics: mean
import NNlib: logsigmoid
using Lux

abstract type LossType end
abstract type MSE <: LossType end
abstract type MAE <: LossType end
abstract type LogLoss <: LossType end
abstract type MLogLoss <: LossType end
abstract type GaussianMLE <: LossType end
abstract type Tweedie <: LossType end

struct MSE_Loss <: Lux.AbstractLossFunction end
(::MSE_Loss)(p::AbstractArray, y) = mean((p .- y) .^ 2)
(::MSE_Loss)(p::AbstractArray, y, w) = sum((p .- y) .^ 2 .* w) / sum(w)
(::MSE_Loss)(p::AbstractArray, y, w, offset) = sum((p .+ offset .- y) .^ 2 .* w) / sum(w)

struct MAE_Loss <: Lux.AbstractLossFunction end
(::MAE_Loss)(p::AbstractArray, y) = mean(abs.(p .- y))
(::MAE_Loss)(p::AbstractArray, y, w) = sum(abs.(p .- y) .* w) / sum(w)
(::MAE_Loss)(p::AbstractArray, y, w, offset) = sum(abs.(p .+ offset .- y) .* w) / sum(w)

struct Log_Loss <: Lux.AbstractLossFunction end
(::Log_Loss)(p::AbstractArray, y) = mean((1 .- y) .* p .- logsigmoid.(p))
(::Log_Loss)(p::AbstractArray, y, w) = sum(w .* ((1 .- y) .* p .- logsigmoid.(p))) / sum(w)
function (::Log_Loss)(p::AbstractArray, y, w, offset)
    p = p .+ offset
    sum(w .* ((1 .- y) .* p .- logsigmoid.(p))) / sum(w)
end

struct Tweedie_Loss{T} <: Lux.AbstractLossFunction
    rho::T
end
Tweedie_Loss() = Tweedie_Loss(1.5f0)
function (l::Tweedie_Loss)(p::AbstractArray, y)
    rho = eltype(p)(l.rho)
    ep = exp.(p)
    mean(2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) - y .* ep .^ (1 - rho) / (1 - rho) + ep .^ (2 - rho) / (2 - rho)))
end
function (l::Tweedie_Loss)(p::AbstractArray, y, w)
    rho = eltype(p)(l.rho)
    ep = exp.(p)
    sum(w .* 2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) - y .* ep .^ (1 - rho) / (1 - rho) + ep .^ (2 - rho) / (2 - rho))) / sum(w)
end
function (l::Tweedie_Loss)(p::AbstractArray, y, w, offset)
    rho = eltype(p)(l.rho)
    ep = exp.(p .+ offset)
    sum(w .* 2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) - y .* ep .^ (1 - rho) / (1 - rho) + ep .^ (2 - rho) / (2 - rho))) / sum(w)
end

struct MLog_Loss <: Lux.AbstractLossFunction end
function (::MLog_Loss)(p::AbstractArray, y)
    y_oh = (UInt32(1):UInt32(size(p, 1))) .== reshape(y, 1, :)
    Lux.CrossEntropyLoss(; logits=Val(true))(p, y_oh)
end
function (::MLog_Loss)(p::AbstractArray, y, w)
    y_oh = (UInt32(1):UInt32(size(p, 1))) .== reshape(y, 1, :)
    per_sample = Lux.CrossEntropyLoss(; logits=Val(true), agg=identity)(p, y_oh)
    sum(vec(per_sample) .* vec(w)) / sum(w)
end
function (::MLog_Loss)(p::AbstractArray, y, w, offset)
    y_oh = (UInt32(1):UInt32(size(p, 1))) .== reshape(y, 1, :)
    per_sample = Lux.CrossEntropyLoss(; logits=Val(true), agg=identity)(p .+ offset, y_oh)
    sum(vec(per_sample) .* vec(w)) / sum(w)
end

_softplus(x) = log(one(x) + exp(x))

struct GaussianMLE_Loss <: Lux.AbstractLossFunction end
function (::GaussianMLE_Loss)(p::AbstractArray, y)
    μ = view(p, 1, :)
    raw_σ = view(p, 2, :)
    y = vec(y)
    T = eltype(p)
    σ = _softplus.(raw_σ) .+ T(1e-4)
    mean(log.(σ) .+ (y .- μ) .^ 2 ./ (2 .* σ .^ 2))
end
function (::GaussianMLE_Loss)(p::AbstractArray, y, w)
    μ = view(p, 1, :)
    raw_σ = view(p, 2, :)
    y = vec(y)
    T = eltype(p)
    σ = _softplus.(raw_σ) .+ T(1e-4)
    sum((log.(σ) .+ (y .- μ) .^ 2 ./ (2 .* σ .^ 2)) .* w) / sum(w)
end
function (::GaussianMLE_Loss)(p::AbstractArray, y, w, offset)
    p_adj = p .+ offset
    μ = view(p_adj, 1, :)
    raw_σ = view(p_adj, 2, :)
    y = vec(y)
    T = eltype(p_adj)
    σ = _softplus.(raw_σ) .+ T(1e-4)
    sum((log.(σ) .+ (y .- μ) .^ 2 ./ (2 .* σ .^ 2)) .* w) / sum(w)
end
const _loss_type_dict = Dict(
    :mse => MSE,
    :mae => MAE,
    :logloss => LogLoss,
    :mlogloss => MLogLoss,
    :gaussian_mle => GaussianMLE,
    :tweedie => Tweedie,
)

get_loss_type(loss::Symbol) = _loss_type_dict[loss]

get_loss_fn(::Type{<:MSE}) = MSE_Loss()
get_loss_fn(::Type{<:MAE}) = MAE_Loss()
get_loss_fn(::Type{<:LogLoss}) = Log_Loss()
get_loss_fn(::Type{<:MLogLoss}) = MLog_Loss()
get_loss_fn(::Type{<:GaussianMLE}) = GaussianMLE_Loss()
get_loss_fn(::Type{<:Tweedie}) = Tweedie_Loss()
get_loss_fn(::Type{<:LossType}) = MSE_Loss() # fallback

get_loss_fn(s::Symbol) = get_loss_fn(get_loss_type(s))

end