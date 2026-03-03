module Metrics

export metric_dict, is_maximise, get_metric

import Statistics: mean, std
import NNlib: logsigmoid, logsoftmax, softmax, relu, hardsigmoid
using Lux

"""
    mse(m, x, y; agg=mean)
    mse(m, x, y, w; agg=mean)
    mse(m, x, y, w, offset; agg=mean)
"""
function mse(m, x, y; agg=mean)
    return agg((vec(m(x)) .- vec(y)) .^ 2)
end
function mse(m, x, y, w; agg=mean)
    return agg((vec(m(x)) .- vec(y)) .^ 2 .* vec(w))
end
function mse(m, x, y, w, offset; agg=mean)
    return agg((vec(m(x)) .+ vec(offset) .- vec(y)) .^ 2 .* vec(w))
end

"""
    mae(m, x, y; agg=mean)
    mae(m, x, y, w; agg=mean)
    mae(m, x, y, w, offset; agg=mean)
"""
function mae(m, x, y; agg=mean)
    return agg(abs.(vec(m(x)) .- vec(y)))
end
function mae(m, x, y, w; agg=mean)
    return agg(abs.(vec(m(x)) .- vec(y)) .* vec(w))
end
function mae(m, x, y, w, offset; agg=mean)
    return agg(abs.(vec(m(x)) .+ vec(offset) .- vec(y)) .* vec(w))
end

"""
    logloss(m, x, y; agg=mean)
    logloss(m, x, y, w; agg=mean)
    logloss(m, x, y, w, offset; agg=mean)
"""
function logloss(m, x, y; agg=mean)
    p = vec(m(x))
    y = vec(y)
    return agg((1 .- y) .* p .- logsigmoid.(p))
end
function logloss(m, x, y, w; agg=mean)
    p = vec(m(x))
    y = vec(y)
    return agg(((1 .- y) .* p .- logsigmoid.(p)) .* vec(w))
end
function logloss(m, x, y, w, offset; agg=mean)
    p = vec(m(x)) .+ vec(offset)
    y = vec(y)
    return agg(((1 .- y) .* p .- logsigmoid.(p)) .* vec(w))
end

"""
    tweedie(m, x, y; agg=mean)
    tweedie(m, x, y, w; agg=mean)
    tweedie(m, x, y, w, offset; agg=mean)
"""
function tweedie(m, x, y; agg=mean)
    rho = eltype(x)(1.5)
    p = exp.(vec(m(x)))
    y = vec(y)
    return agg(2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* p .^ (1 - rho) / (1 - rho) .+
                     p .^ (2 - rho) / (2 - rho)))
end
function tweedie(m, x, y, w; agg=mean)
    rho = eltype(x)(1.5)
    p = exp.(vec(m(x)))
    y = vec(y)
    w = vec(w)
    return agg(w .* 2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* p .^ (1 - rho) / (1 - rho) .+
                          p .^ (2 - rho) / (2 - rho)))
end
function tweedie(m, x, y, w, offset; agg=mean)
    rho = eltype(x)(1.5)
    p = exp.(vec(m(x)) .+ vec(offset))
    y = vec(y)
    w = vec(w)
    return agg(w .* 2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* p .^ (1 - rho) / (1 - rho) .+
                          p .^ (2 - rho) / (2 - rho)))
end

"""
    mlogloss(m, x, y; agg=mean)
    mlogloss(m, x, y, w; agg=mean)
    mlogloss(m, x, y, w, offset; agg=mean)
"""
function mlogloss(m, x, y; agg=mean)
    p = m(x)
    k = size(p, 1)
    y_oh = (UInt32(1):UInt32(k)) .== reshape(y, 1, :)
    lsm = logsoftmax(p; dims=1)
    return agg(vec(-sum(y_oh .* lsm; dims=1)))
end
function mlogloss(m, x, y, w; agg=mean)
    p = m(x)
    k = size(p, 1)
    y_oh = (UInt32(1):UInt32(k)) .== reshape(y, 1, :)
    lsm = logsoftmax(p; dims=1)
    return agg(vec(-sum(y_oh .* lsm; dims=1)) .* vec(w))
end
function mlogloss(m, x, y, w, offset; agg=mean)
    p = m(x) .+ offset
    k = size(p, 1)
    y_oh = (UInt32(1):UInt32(k)) .== reshape(y, 1, :)
    lsm = logsoftmax(p; dims=1)
    return agg(vec(-sum(y_oh .* lsm; dims=1)) .* vec(w))
end

"""
    gaussian_mle(m, x, y; agg=mean)
    gaussian_mle(m, x, y, w; agg=mean)
    gaussian_mle(m, x, y, w, offset; agg=mean)
"""
_gaussian_mle_elt(μ, σ, y) =
    -σ - (y - μ)^2 / (2 * max(oftype(σ, 2e-7), exp(2 * σ)))

_gaussian_mle_elt(μ, σ, y, w) =
    (-σ - (y - μ)^2 / (2 * max(oftype(σ, 2e-7), exp(2 * σ)))) * w

function gaussian_mle(m, x, y; agg=mean)
    p = m(x)
    metric = agg(_gaussian_mle_elt.(view(p, 1, :), view(p, 2, :), vec(y)))
    return metric
end
function gaussian_mle(m, x, y, w; agg=mean)
    p = m(x)
    metric = agg(_gaussian_mle_elt.(view(p, 1, :), view(p, 2, :), vec(y), vec(w)))
    return metric
end
function gaussian_mle(m, x, y, w, offset; agg=mean)
    p = m(x) .+ offset
    metric = agg(_gaussian_mle_elt.(view(p, 1, :), view(p, 2, :), vec(y), vec(w)))
    return metric
end

function get_metric(ts, data, eval_compiled)
    metric = 0.0f0
    ws = 0.0f0
    st = Lux.testmode(ts.states)
    for d in data
        if length(d) == 2
            m_val, w_val = eval_compiled(d[1], d[2], ts.parameters, st)
        elseif length(d) == 3
            m_val, w_val = eval_compiled(d[1], d[2], d[3], ts.parameters, st)
        else
            m_val, w_val = eval_compiled(d[1], d[2], d[3], d[4], ts.parameters, st)
        end
        metric += Float32(m_val)
        ws += Float32(w_val)
    end
    return Float64(metric / ws)
end

const metric_dict = Dict(
    :mse => mse,
    :mae => mae,
    :logloss => logloss,
    :mlogloss => mlogloss,
    :gaussian_mle => gaussian_mle,
    :tweedie => tweedie,
)

is_maximise(::typeof(mse)) = false
is_maximise(::typeof(mae)) = false
is_maximise(::typeof(logloss)) = false
is_maximise(::typeof(mlogloss)) = false
is_maximise(::typeof(gaussian_mle)) = true
is_maximise(::typeof(tweedie)) = false

end