module ModernNCA

export ModernNCAConfig

import ..Models
import ..Models: Architecture, NeuroTabModel
import ..Losses: LossType, MSE, MAE, LogLoss, MLogLoss

using Lux
using Lux: Functors
using LuxCore
using NNlib: relu
using ChainRulesCore: ChainRulesCore, RuleConfig, HasReverseMode, NoTangent, ZeroTangent,
    rrule_via_ad, unthunk
using Random: randperm, AbstractRNG
using StatsBase: sample
using DataFrames: AbstractDataFrame, select
using CategoricalArrays

"""
    ModernNCAConfig(; d_embedding=128, n_blocks=2, d_block=256,
                     dropout=0.1, temperature=1.0, sample_rate=0.8,
                     corpus_chunk_size=2048, eps=1f-8)

Hyperparameters for ModernNCA. Pass as the `arch` argument to `NeuroTabRegressor`
or `NeuroTabClassifier`.

# Arguments
- `d_embedding`: encoder output dimension.
- `n_blocks`, `d_block`: number and hidden width of MLP blocks after the linear layer.
- `dropout`: dropout rate inside each block; skipped when `<= 0`.
- `temperature`: softmax temperature on negative pairwise distances.
- `sample_rate`: fraction of the batch complement sampled as candidates per
  training step (Stochastic Neighborhood Sampling). `1.0` uses the full complement.
- `corpus_chunk_size`: keys encoded and attended per chunk, in training and
  inference. This is the memory knob.
- `eps`: numerical floor under `sqrt` and as a `temperature` lower bound.
"""
struct ModernNCAConfig <: Architecture
    d_embedding::Int
    n_blocks::Int
    d_block::Int
    dropout::Float32
    temperature::Float32
    sample_rate::Float32
    corpus_chunk_size::Int
    eps::Float32
end

function ModernNCAConfig(;
    d_embedding::Int=128, n_blocks::Int=2, d_block::Int=256,
    dropout::Real=0.1, temperature::Real=1.0, sample_rate::Real=0.8,
    corpus_chunk_size::Int=2048, eps::Real=1.0f-8,
)
    corpus_chunk_size > 0 || throw(ArgumentError("corpus_chunk_size must be positive"))
    return ModernNCAConfig(d_embedding, n_blocks, d_block, Float32(dropout),
        Float32(temperature), Float32(sample_rate), corpus_chunk_size, Float32(eps))
end

include("model.jl")

"""
    (cfg::ModernNCAConfig)(; ins, outsize, loss_type=MSE)

Build a `ModernNCAModel` with a `NoOpLayer` embedding. Prefer
`Models.build_chain` so the real feature embedding lives inside the model.

# Arguments
- `ins`: number of input features after embedding (or raw features if `NoOpLayer`).
- `outsize`: output size (`1` for MSE/MAE/LogLoss, `K` for MLogLoss).
- `loss_type`: `MSE`, `MAE`, `LogLoss`, or `MLogLoss`.
"""
(cfg::ModernNCAConfig)(; ins, outsize, loss_type::Type{<:LossType}=MSE) =
    _build_model(cfg, NoOpLayer(), ins, outsize, loss_type)

"""
    Models.build_chain(cfg::ModernNCAConfig, embedding; ins, outsize, loss_type)

Build `ModernNCAModel` with `embedding` inside the Lux container so query and
corpus rows share one encoder.

# Arguments
- `cfg`: ModernNCA config.
- `embedding`: feature-embedding layer.
- `ins`: embedding output dimension (encoder input size).
- `outsize`: output size (`1` for MSE/MAE/LogLoss, `K` for MLogLoss).
- `loss_type`: `MSE`, `MAE`, `LogLoss`, or `MLogLoss`.
"""
Models.build_chain(cfg::ModernNCAConfig, embedding; ins, outsize, loss_type, kwargs...) =
    _build_model(cfg, embedding, ins, outsize, loss_type)

"""
    ModernNCALoader

Training iterator yielding `((x, cand_x, cand_y, y), y)`. Query rows follow a
shuffled epoch permutation; `n_cand` candidates are resampled every step from
the complement of the query batch. The corpus lives on device.

# Arguments
- `full_x`: feature matrix `(ins, N)` on device.
- `full_y`: encoded targets of length `N`.
- `batchsize`: query rows per step.
- `n_cand`: candidate rows sampled from the batch complement.
- `rng`: sampler for the epoch permutation and candidate indices.
- `dev`: device for index tensors.
"""
struct ModernNCALoader{X,Y,R<:AbstractRNG,D}
    full_x::X
    full_y::Y
    batchsize::Int
    n_cand::Int
    rng::R
    dev::D
end

Base.length(l::ModernNCALoader) = fld(size(l.full_x, 2), l.batchsize)

"""
    Base.iterate(l::ModernNCALoader, state=nothing)

One training step. `state = (perm, start)`. Candidates are drawn from `perm`
outside the window `start:stop` with an index skip, so the complement is never
materialised.

# Arguments
- `l`: loader.
- `state`: `(perm, start)` after the first call; `nothing` starts a new epoch.
"""
function Base.iterate(l::ModernNCALoader, state=nothing)
    n = size(l.full_x, 2)
    perm, start = state === nothing ? (randperm(l.rng, n), 1) : state
    stop = start + l.batchsize - 1
    stop > n && return nothing

    bidx = l.dev(perm[start:stop])
    x, y = l.full_x[:, bidx], l.full_y[bidx]

    if l.n_cand > 0
        js = sample(l.rng, 1:(n - l.batchsize), l.n_cand; replace=false)
        cidx = l.dev(perm[@. ifelse(js < start, js, js + l.batchsize)])
        cand_x, cand_y = l.full_x[:, cidx], l.full_y[cidx]
    else
        cand_x, cand_y = similar(l.full_x, size(l.full_x, 1), 0), similar(l.full_y, 0)
    end
    return ((x, cand_x, cand_y, y), y), (perm, stop + 1)
end

"""
    build_corpus(df, feature_names, target_name, loss_type, scalers) -> (full_x, full_y)

Feature matrix `(ins, N)` as `Float32` and encoded targets.

# Arguments
- `df`: training data frame.
- `feature_names`: columns used as features.
- `target_name`: target column.
- `loss_type`: `MSE`, `MAE`, `LogLoss`, or `MLogLoss`.
- `scalers`: target mean/std for MSE/MAE, or `nothing`.
"""
function build_corpus(df::AbstractDataFrame, feature_names, target_name,
    loss_type::Type{<:LossType}, scalers)
    full_x = permutedims(Matrix{Float32}(select(df, collect(feature_names))))
    return full_x, _encode_targets(df, target_name, loss_type, scalers)
end

"""
    _encode_targets(df, target_name, loss_type, scalers)

MLogLoss: 1-based `UInt32` codes. LogLoss: `Float32` in `{0, 1}`. MSE/MAE:
`Float32`, scaled.
"""
_encode_targets(df, target_name, ::Type{<:MLogLoss}, _) =
    UInt32.(CategoricalArrays.levelcode.(df[!, target_name]))

function _encode_targets(df, target_name, ::Type{<:LogLoss}, _)
    col = df[!, target_name]
    eltype(col) <: CategoricalValue || return Float32.(col)
    length(CategoricalArrays.levels(col)) == 2 ||
        error("For `loss=:logloss`, target must have exactly 2 classes.")
    return Float32.(CategoricalArrays.levelcode.(col) .- 1)
end

function _encode_targets(df, target_name, ::Type{<:Union{MSE,MAE}}, scalers)
    y = Float32.(df[!, target_name])
    isnothing(scalers) || (y .= (y .- scalers.mu) ./ scalers.sigma)
    return y
end

"""
    Models.train_dataloader(cfg::ModernNCAConfig, ...)

Build `ModernNCALoader` and stash the raw corpus on `m.info[:nca_ref]`.
`n_cand = floor(sample_rate × (N − batchsize))`. `sample_rate ≥ 1` uses the
full complement; `batchsize == N` gives `n_cand = 0`. Only the Zygote backend
is supported. Weights, offsets, and group padding are rejected.

# Arguments
- `cfg`: ModernNCA config.
- `m`: fitted model wrapper.
- `df`: training data frame.
- `feature_names`
- `target_name`
- `loss_type`
- `scalers`
- `batchsize`
- `dev`: device
- `rng`
"""
function Models.train_dataloader(cfg::ModernNCAConfig, m::NeuroTabModel, ::Any, df;
    feature_names, target_name, loss_type, scalers, batchsize, dev, rng,
    weight_name=nothing, offset_name=nothing, group_key=nothing, backend=:zygote, kwargs...)
    backend == :zygote ||
        throw(ArgumentError("ModernNCA training supports only the Zygote backend (got $backend)"))
    for (val, name) in
        ((weight_name, :weight_name), (offset_name, :offset_name), (group_key, :group_key))
        isnothing(val) || throw(ArgumentError("ModernNCA does not support `$name`"))
    end
    cx, cy = build_corpus(df, feature_names, target_name, loss_type, scalers)
    m.info[:nca_ref] = (cx=cx, cy=cy)
    n = size(cx, 2)
    batchsize = min(batchsize, n)
    pool = n - batchsize
    n_cand = pool == 0 ? 0 :
             cfg.sample_rate >= 1.0f0 ? pool :
             max(Int(floor(cfg.sample_rate * pool)), 1)
    return ModernNCALoader(dev(cx), dev(cy), batchsize, n_cand, rng, dev)
end

function _corpus(m::ModernNCAModel, info, dev)
    ref = info[:nca_ref]
    return Corpus(dev(ref.cx), _target_layout(m.loss_type, dev(ref.cy)), info)
end

"""
    Models.eval_dataloader(m::ModernNCAModel, info, data, dev, ps, st)

Attach a [`Corpus`](@ref) to every eval batch `(x, y, ...)`; the encoding
refreshes per round.

# Arguments
- `m`: ModernNCA model.
- `info`: fit metadata; must contain `:nca_ref` and `:nrounds`.
- `data`: default eval iterator of `(x, y, ...)`.
- `dev`: device.
- `ps`, `st`: unused; encoding happens on the first forward of the round.
"""
function Models.eval_dataloader(m::ModernNCAModel, info, data, dev, ::Any, ::Any)
    corpus = _corpus(m, info, dev)
    return Iterators.map(d -> ((d[1], corpus), Base.tail(d)...), data)
end

"""
    Models.infer_dataloader(m::ModernNCAModel, info, data, dev, ps, st)

Attach a [`Corpus`](@ref) to every inference batch; it is encoded once, on the
first batch. With `grouped=true`, preserve each batch's row mask. Reactant is
rejected.

# Arguments
- `m`: ModernNCA model.
- `info`: fit metadata; must contain `:nca_ref` and `:nrounds`.
- `data`: default inference iterator.
- `dev`: device.
- `ps`, `st`: unused; encoding happens on the first forward.
- `backend`: AD backend; `:reactant` throws.
- `grouped`: if `true`, keep each batch's row mask.
"""
function Models.infer_dataloader(m::ModernNCAModel, info, data, dev, ::Any, ::Any;
    backend=:zygote, grouped::Bool=false)
    backend == :reactant &&
        throw(ArgumentError("ModernNCA does not support the Reactant backend"))
    corpus = _corpus(m, info, dev)
    return grouped ?
           Iterators.map(d -> ((d[1], corpus), d[2]), data) :
           Iterators.map(x -> (x, corpus), data)
end

end
