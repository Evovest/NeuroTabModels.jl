module Metrics

export metric_dict, is_maximise, get_metric

import Statistics: mean, std
import StatsBase: tiedrank, denserank
import NNlib: logsigmoid, logsoftmax, softmax, relu, hardsigmoid
import ..Losses: _pearson_value
using Lux

"""
    percent_rank(x::AbstractVector)

Transform input into percentile (uniformly distributed between 0-1)
"""
function percent_rank(x::AbstractVector)
    result = fill(NaN, length(x))
    idx = findall(!isnan, x)
    isempty(idx) && return result
    @views result[idx] .= tiedrank(view(x, idx)) ./ (length(idx) + 1)
    return result
end

# One target is scored on flat vectors, since broadcasting over a `(1, B)` matrix loses SIMD.
# `_flat` returns a vector or a matrix, so `_score` passes it through a function barrier.
_flat(p) = size(p, 1) == 1 ? vec(p) : p
_score(f, p) = f(_flat(p))

# Lay `a` out like the prediction `p`: flat for one target, or one column per observation of a
# `(T, B)` prediction, where targets keep their rows and per-row weights and a shared offset
# become a `(1, B)` row broadcast over targets.
_obs(a, p::AbstractVector) = vec(a)
_obs(a, p) = reshape(a, :, size(p, 2))

"""
    mse(m, x, y; agg=mean)
    mse(m, x, y, w; agg=mean)
    mse(m, x, y, w, offset; agg=mean)
"""
function mse(m, x, y; agg=mean)
    return _score(p -> agg((p .- _obs(y, p)) .^ 2), m(x))
end
function mse(m, x, y, w; agg=mean)
    return _score(p -> agg((p .- _obs(y, p)) .^ 2 .* _obs(w, p)), m(x))
end
function mse(m, x, y, w, offset; agg=mean)
    return _score(p -> agg((p .+ _obs(offset, p) .- _obs(y, p)) .^ 2 .* _obs(w, p)), m(x))
end

"""
    mae(m, x, y; agg=mean)
    mae(m, x, y, w; agg=mean)
    mae(m, x, y, w, offset; agg=mean)
"""
function mae(m, x, y; agg=mean)
    return _score(p -> agg(abs.(p .- _obs(y, p))), m(x))
end
function mae(m, x, y, w; agg=mean)
    return _score(p -> agg(abs.(p .- _obs(y, p)) .* _obs(w, p)), m(x))
end
function mae(m, x, y, w, offset; agg=mean)
    return _score(p -> agg(abs.(p .+ _obs(offset, p) .- _obs(y, p)) .* _obs(w, p)), m(x))
end

"""
    logloss(m, x, y; agg=mean)
    logloss(m, x, y, w; agg=mean)
    logloss(m, x, y, w, offset; agg=mean)
"""
_logloss(p, y) = (1 .- y) .* p .- logsigmoid.(p)

function logloss(m, x, y; agg=mean)
    return _score(p -> agg(_logloss(p, _obs(y, p))), m(x))
end
function logloss(m, x, y, w; agg=mean)
    return _score(p -> agg(_logloss(p, _obs(y, p)) .* _obs(w, p)), m(x))
end
function logloss(m, x, y, w, offset; agg=mean)
    return _score(m(x)) do p
        p = p .+ _obs(offset, p)
        agg(_logloss(p, _obs(y, p)) .* _obs(w, p))
    end
end

"""
    tweedie(m, x, y; agg=mean)
    tweedie(m, x, y, w; agg=mean)
    tweedie(m, x, y, w, offset; agg=mean)
"""
function _tweedie(p, y, rho)
    return 2 .* (y .^ (2 - rho) / (1 - rho) / (2 - rho) .- y .* p .^ (1 - rho) / (1 - rho) .+ p .^ (2 - rho) / (2 - rho))
end

function tweedie(m, x, y; agg=mean)
    rho = eltype(x)(1.5)
    return _score(p -> agg(_tweedie(exp.(p), _obs(y, p), rho)), m(x))
end
function tweedie(m, x, y, w; agg=mean)
    rho = eltype(x)(1.5)
    return _score(p -> agg(_obs(w, p) .* _tweedie(exp.(p), _obs(y, p), rho)), m(x))
end
function tweedie(m, x, y, w, offset; agg=mean)
    rho = eltype(x)(1.5)
    return _score(p -> agg(_obs(w, p) .* _tweedie(exp.(p .+ _obs(offset, p)), _obs(y, p), rho)), m(x))
end

# offset in the (K, B) layout of `m(x)`: a vector or a grouped (1, 1, B) for one row,
# a (K, B) matrix or a grouped (K, 1, B) for K
_offset_2d(offset::AbstractVector) = reshape(offset, 1, :)
_offset_2d(offset::AbstractMatrix) = offset
_offset_2d(offset::AbstractArray{T,3}) where {T} = reshape(offset, size(offset, 1), :)

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
    p = m(x) .+ _offset_2d(offset)
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
_gaussian_mle_elt(μ, σ, y) = -σ - (y - μ)^2 / (2 * max(oftype(σ, 2e-7), exp(2 * σ)))

_gaussian_mle_elt(μ, σ, y, w) = (-σ - (y - μ)^2 / (2 * max(oftype(σ, 2e-7), exp(2 * σ)))) * w

# Rows interleave per target, as in EvoTrees: odd rows are μ and even rows log-σ.
# One target takes its two rows as vectors, which broadcast faster than `(1, B)` views.
function _mu_logsigma(p)
    size(p, 1) == 2 && return view(p, 1, :), view(p, 2, :)
    return view(p, 1:2:size(p, 1), :), view(p, 2:2:size(p, 1), :)
end

function gaussian_mle(m, x, y; agg=mean)
    μ, σ = _mu_logsigma(m(x))
    return agg(_gaussian_mle_elt.(μ, σ, _obs(y, μ)))
end
function gaussian_mle(m, x, y, w; agg=mean)
    μ, σ = _mu_logsigma(m(x))
    return agg(_gaussian_mle_elt.(μ, σ, _obs(y, μ), _obs(w, μ)))
end
function gaussian_mle(m, x, y, w, offset; agg=mean)
    μ, σ = _mu_logsigma(m(x) .+ _offset_2d(offset))
    return agg(_gaussian_mle_elt.(μ, σ, _obs(y, μ), _obs(w, μ)))
end

"""
    pearson(m, x, y; agg=mean)
    pearson(m, x, y, w; agg=mean)
    pearson(m, x, y, w, offset; agg=mean)

Uses the first output (`μ` when `gaussian_mle` returns `size(p, 1) == 2`).
A flat group (constant predictions or target, or a single row) scores 0 and keeps its weight.
With several targets each correlates on its own and the metric is their mean, as in EvoTrees.
"""
# Target `t` reads output row `t`, or its μ in row `2t - 1` when there are two outputs per
# target, as with `gaussian_mle`. As in EvoTrees, the layout is read off the shapes.
function _corr_pred(p, y, t=1)
    stride = size(p, 1) == 2 * size(y, 1) ? 2 : 1
    return vec(view(p, stride * (t - 1) + 1, :))
end

function _pearson_mean(p, y, w)
    T = size(y, 1)
    T == 1 && return _pearson_value(_corr_pred(p, y), y, w)
    return sum(t -> _pearson_value(_corr_pred(p, y, t), selectdim(y, 1, t), w), 1:T) / T
end

# Scaled by the eval step's denominator, which counts every target, so the logged value is the mean.
function pearson(m, x, y; agg=mean)
    p = m(x)
    return _pearson_mean(p, y, one.(_corr_pred(p, y))) * length(y)
end
function pearson(m, x, y, w; agg=mean)
    return _pearson_mean(m(x), y, w) * sum(w) * size(y, 1)
end
function pearson(m, x, y, w, offset; agg=mean)
    return _pearson_mean(m(x) .+ _offset_2d(offset), y, w) * sum(w) * size(y, 1)
end

"""
    s_corr(m, x, y; agg=mean)
    s_corr(m, x, y, w; agg=mean)
    s_corr(m, x, y, w, offset; agg=mean)
"""
function s_corr(m, x, y, w; agg=mean)
    p = vec(view(m(x), 1, :))
    y = vec(y)
    w = vec(w)

    p = sortperm(sortperm(p)) ./ length(p)
    y = sortperm(sortperm(y)) ./ length(y)

    p_mean = w' * p / sum(w)
    p_var = w' * (p .^ 2) / sum(w) - p_mean^2
    y_mean = w' * y / sum(w)
    y_var = w' * (y .^ 2) / sum(w) - y_mean^2
    py_mean = w' * (p .* y) / sum(w)
    return (py_mean - p_mean * y_mean) / (sqrt(p_var) * sqrt(y_var)) * sum(w)
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
    :pearson => pearson,
)

is_maximise(::typeof(mse)) = false
is_maximise(::typeof(mae)) = false
is_maximise(::typeof(logloss)) = false
is_maximise(::typeof(mlogloss)) = false
is_maximise(::typeof(gaussian_mle)) = true
is_maximise(::typeof(tweedie)) = false
is_maximise(::typeof(pearson)) = true

end