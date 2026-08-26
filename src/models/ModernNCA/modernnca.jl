module ModernNCA

export ModernNCAConfig

import ..Models
import ..Models: Architecture, NeuroTabModel
import ..Losses: LossType, MSE, MAE, LogLoss, MLogLoss, GaussianMLE

using Lux
using LuxCore
using NNlib: relu, softmax
using Random: randperm, AbstractRNG
using DataFrames: AbstractDataFrame, select
using CategoricalArrays

"""
    ModernNCAConfig(; d_embedding=128, n_blocks=2, d_block=256,
                     dropout=0.1, temperature=1.0, sample_rate=0.8,
                     max_candidates=8192, eps=1f-8)

Hyperparameters for ModernNCA. Pass as the `arch` argument to `NeuroTabRegressor`
or `NeuroTabClassifier`.

# Arguments
- `d_embedding`: encoder output dimension.
- `n_blocks`, `d_block`: number and hidden width of post-encoder MLP blocks.
- `dropout`: dropout rate inside each block; skipped when `<= 0`.
- `temperature`: softmax temperature on negative pairwise distances.
- `sample_rate`: fraction of the batch complement sampled as candidates per step
  (Stochastic Neighborhood Sampling). `1.0` uses the full complement.
- `max_candidates`: cap on sampled training candidates per batch. Set `<= 0`
  to disable the cap.
- `eps`: numerical floor in `sqrt` and as a `temperature` lower bound.
"""
struct ModernNCAConfig <: Architecture
    d_embedding::Int
    n_blocks::Int
    d_block::Int
    dropout::Float32
    temperature::Float32
    sample_rate::Float32
    max_candidates::Int
    eps::Float32
end

"""
    ModernNCAModel

Lux wrapper holding the backbone encoder, config, output size, and loss type.
"""
struct ModernNCAModel{B,LT} <: LuxCore.AbstractLuxWrapperLayer{:backbone}
    backbone::B
    cfg::ModernNCAConfig
    outsize::Int
    loss_type::Type{LT}
end

"""
    ModernNCALoader

Training iterator yielding `((x, cand_x, cand_y, y), y)` batches. The corpus is
moved to device once at construction; query rows follow a shuffled epoch
permutation and candidates are resampled each step.

# Fields
- `full_x`, `full_y`: device corpus.
- `batchsize`: query rows per step.
- `n_cand`: candidate rows sampled per step (`0` when `batchsize == N`).
- `rng`: RNG for sampling.
- `dev`: device-transfer callable.
"""
struct ModernNCALoader{X,Y,R<:AbstractRNG,D}
    full_x::X
    full_y::Y
    batchsize::Int
    n_cand::Int
    rng::R
    dev::D
end

function ModernNCAConfig(;
    d_embedding::Int=128,
    n_blocks::Int=2,
    d_block::Int=256,
    dropout::Real=0.1,
    temperature::Real=1.0,
    sample_rate::Real=0.8,
    max_candidates::Int=8192,
    eps::Real=1.0f-8,
)
    return ModernNCAConfig(
        d_embedding,
        n_blocks,
        d_block,
        Float32(dropout),
        Float32(temperature),
        Float32(sample_rate),
        max_candidates,
        Float32(eps),
    )
end

"""
    _backbone(cfg, ins, embedding_layer)

Build the ModernNCA encoder: embedding, linear, then n_blocks times (BN, Dense(relu), Dropout, Dense), then BN.
"""
function _backbone(cfg::ModernNCAConfig, ins::Int)
    layers = Any[Dense(ins => cfg.d_embedding)]
    for _ in 1:cfg.n_blocks
        push!(layers, BatchNorm(cfg.d_embedding))
        push!(layers, Dense(cfg.d_embedding => cfg.d_block, relu))
        cfg.dropout > 0 && push!(layers, Dropout(cfg.dropout))
        push!(layers, Dense(cfg.d_block => cfg.d_embedding))
    end
    cfg.n_blocks > 0 && push!(layers, BatchNorm(cfg.d_embedding))
    return Chain(layers...)
end

function (cfg::ModernNCAConfig)(; ins, outsize, loss_type::Type{<:LossType}=MSE)
    return ModernNCAModel(_backbone(cfg, ins), cfg, Int(outsize), loss_type)
end

Base.length(l::ModernNCALoader) = fld(size(l.full_x, 2), l.batchsize)

"""
    _pairwise_dist(q, k, ϵ) -> Matrix

`(num_keys, batch)` Euclidean distance matrix between `q` `(d, batch)` and `k` `(d, num_keys)`.
`ϵ` is added under `sqrt` for numerical stability.
"""
function _pairwise_dist(q::AbstractMatrix, k::AbstractMatrix, ϵ::Float32)
    q2 = sum(abs2, q; dims=1)  # (1, batch)
    k2 = sum(abs2, k; dims=1)  # (1, num_keys)
    kq = k' * q                 # (num_keys, batch)
    return sqrt.(max.(0.0f0, k2' .+ q2 .- 2.0f0 .* kq) .+ ϵ)
end

_diag_inf(i, j, d) = ifelse(i == j, typemax(typeof(d)), d)

"""
    _mask_diag(d) -> Matrix

Set the diagonal of distance matrix `d` to `typemax`, collapsing self-attention
weights to zero during training.
"""
_mask_diag(d::AbstractMatrix) = _diag_inf.(reshape(1:size(d, 1), :, 1), reshape(1:size(d, 2), 1, :), d)

_to_output(::Type{<:Union{MSE,MAE}}, α, cy, _) = reshape(α' * cy, 1, :)

function _to_output(::Type{<:LogLoss}, α, cy, _)
    p = clamp.(reshape(α' * cy, 1, :), 1.0f-6, 1.0f0 - 1.0f-6)
    return log.(p ./ (1.0f0 .- p))
end

function _to_output(::Type{<:MLogLoss}, α, cy, outsize::Int)
    oh = ((k, c) -> ifelse(k == c, 1.0f0, 0.0f0)).(reshape(UInt32(1):UInt32(outsize), :, 1), reshape(cy, 1, :))
    return log.(clamp.(oh * α, 1.0f-7, Inf32))
end

"""
    _nca_logits(m, zq, zk, cy; mask_self=false) -> Matrix

Softmax attention over corpus keys `zk` for query embeddings `zq`.
When `mask_self=true`, diagonal entries are masked (training path).
"""
function _nca_logits(m::ModernNCAModel, zq, zk, cy; mask_self::Bool=false)
    d = _pairwise_dist(zq, zk, m.cfg.eps) ./ max(m.cfg.temperature, m.cfg.eps)
    mask_self && (d = _mask_diag(d))
    α = softmax(-d; dims=1)
    return _to_output(m.loss_type, α, cy, m.outsize)
end

"""
    _encode_corpus(m, cx, ps, st; chunk=2048) -> Matrix

Encode raw corpus `cx` `(d_ins, N)` in chunks of `chunk` rows to bound peak
memory and stay within BatchNorm shape limits. Returns `zk` `(d_embedding, N)`.
"""
function _encode_corpus(m::ModernNCAModel, cx, ps, st; chunk::Int=2048)
    n = size(cx, 2)
    if n <= chunk
        return m.backbone(cx, ps, st)[1]
    end
    z0, _ = m.backbone(cx[:, 1:chunk], ps, st)
    zk = similar(z0, size(z0, 1), n)
    zk[:, 1:chunk] .= z0
    s = chunk + 1
    while s <= n
        e = min(s + chunk - 1, n)
        zk[:, s:e] .= m.backbone(cx[:, s:e], ps, st)[1]
        s = e + 1
    end
    return zk
end

"""
    (m::ModernNCAModel)((x, cand_x, cand_y, y), ps, st)

Training forward: encodes queries and candidates separately, matching the
reference ModernNCA BatchNorm behavior. The encoded queries are prepended to
the candidate embeddings before NCA attention, then self-neighbors are masked.
"""
function (m::ModernNCAModel)((x, cand_x, cand_y, y)::Tuple{Any,Any,Any,Any}, ps, st)
    zq, st_bb = m.backbone(x, ps, st)
    zc, st_bb = size(cand_x, 2) == 0 ? (similar(zq, size(zq, 1), 0), st_bb) : m.backbone(cand_x, ps, st_bb)
    z_all = hcat(zq, zc)
    cy = vcat(vec(y), vec(cand_y))
    return _nca_logits(m, zq, z_all, cy; mask_self=true), st_bb
end

"""
    (m::ModernNCAModel)((x, cx, cy), ps, st)

Inference/eval forward: encodes queries and the raw corpus `cx` with the current
parameters, then attends over the corpus. Encoding is done here (not pre-computed)
so the corpus embeddings always match the current model during evaluation.
"""
function (m::ModernNCAModel)((x, cand_x, cand_y)::Tuple{Any,Any,Any}, ps, st)
    zq, st_bb = m.backbone(x, ps, st)
    zk = _encode_corpus(m, cand_x, ps, st_bb)
    return _nca_logits(m, zq, zk, cand_y), st_bb
end

"""
    Base.iterate(l::ModernNCALoader, state=nothing)

`state = (perm, start)` carries the epoch permutation across calls, so query
batches are random without depending on dataframe row order.

Candidates are sampled from the complement of the query batch within `perm`.
The index-skip mapping avoids allocating that complement: for `j` in
`1:(n-batchsize)`, use `perm[j]` before the batch window and
`perm[j + batchsize]` after it. `n_cand == 0` only when `batchsize == n`; then
the forward pass keys on the batch itself with diagonal masking.
"""
function Base.iterate(l::ModernNCALoader, state=nothing)
    n = size(l.full_x, 2)
    perm, start = state === nothing ? (randperm(l.rng, n), 1) : state
    stop = start + l.batchsize - 1
    stop > n && return nothing

    batch_idx = perm[start:stop]

    bidx = l.dev(batch_idx)
    x = l.full_x[:, bidx]
    y = l.full_y[bidx]

    if l.n_cand > 0
        m = n - l.batchsize
        js = rand(l.rng, 1:m, l.n_cand)
        cidx = l.dev(perm[@. ifelse(js < start, js, js + l.batchsize)])
        cand_x = l.full_x[:, cidx]
        cand_y = l.full_y[cidx]
    else
        cand_x = similar(l.full_x, size(l.full_x, 1), 0)
        cand_y = similar(l.full_y, 0)
    end
    return ((x, cand_x, cand_y, y), y), (perm, stop + 1)
end

"""
    build_corpus(df, feature_names, target_name, loss_type, scalers)

Return `(full_x, full_y)`: the corpus feature matrix `(ins, N)` as `Float32`
and the encoded target vector, ready for `ModernNCALoader`.
"""
function build_corpus(df::AbstractDataFrame, feature_names, target_name, loss_type::Type{<:LossType}, scalers)
    full_x = permutedims(Matrix{Float32}(select(df, collect(feature_names))))
    full_y = _encode_targets(df, target_name, loss_type, scalers)
    return full_x, full_y
end

"""
    _encode_targets(df, target_name, loss_type, scalers)

Encode targets for each loss type:
- `MLogLoss`: 1-based `UInt32` class codes.
- `LogLoss`: `Float32` in `{0, 1}`.
- `MSE / MAE`: `Float32`, standardised when `scalers` is provided.

# Arguments
- `df`: source data frame.
- `target_name`: target column.
- `loss_type`: target encoding dispatch.
- `scalers`: optional target scaler `(mu, sigma)`.
"""
_encode_targets(df, target_name, ::Type{<:MLogLoss}, _) = UInt32.(CategoricalArrays.levelcode.(df[!, target_name]))

function _encode_targets(df, target_name, ::Type{<:LogLoss}, _)
    col = df[!, target_name]
    eltype(col) <: CategoricalValue || return Float32.(col)
    levels = CategoricalArrays.levels(col)
    length(levels) == 2 || error("For `loss=:logloss`, target must have exactly 2 classes.")
    return Float32.(CategoricalArrays.levelcode.(col) .- 1)
end

function _encode_targets(df, target_name, ::Type{<:Union{MSE,MAE,GaussianMLE}}, scalers)
    y = Float32.(df[!, target_name])
    isnothing(scalers) || (y .= (y .- scalers.mu) ./ scalers.sigma)
    return y
end

# """
#     Models.build_chain(cfg::ModernNCAConfig; ins, outsize, loss_type, kwargs...)

# Pass the embedding layer into the backbone instead of wrapping it in an outer `Chain`.
# """
# Models.build_chain(cfg::ModernNCAConfig;
#     ins, outsize, loss_type, kwargs...) =
#     cfg(; ins, outsize, loss_type)

"""
    Models.train_dataloader(cfg::ModernNCAConfig, ...)

Build `ModernNCALoader` and stash the raw corpus on `m.info[:nca_ref]`.
`n_cand = floor(sample_rate × (N − batchsize))`, capped by
`cfg.max_candidates` when positive. `sample_rate ≥ 1` uses the full complement
before the cap; `batchsize == N` gives `n_cand = 0`.

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
function Models.train_dataloader(
    cfg::ModernNCAConfig,
    m::NeuroTabModel,
    ::Any,
    df;
    feature_names,
    target_name,
    loss_type,
    scalers,
    batchsize,
    dev,
    rng,
    kwargs...,
)
    cx, cy = build_corpus(df, feature_names, target_name, loss_type, scalers)
    m.info[:nca_ref] = (cx=cx, cy=cy)
    n = size(cx, 2)
    batchsize = min(batchsize, n)
    pool = n - batchsize
    raw_n_cand = if pool == 0
        0
    elseif cfg.sample_rate >= 1.0f0
        pool
    else
        max(Int(floor(cfg.sample_rate * pool)), 1)
    end
    n_cand = cfg.max_candidates > 0 ? min(raw_n_cand, cfg.max_candidates) : raw_n_cand
    return ModernNCALoader(dev(cx), dev(cy), batchsize, n_cand, rng, dev)
end

"""
    Models.infer_dataloader(m::ModernNCAModel, ...)

Wrap each inference batch as `(x, cx, cy)` with the raw training corpus.
The corpus is encoded inside the forward pass using the current parameters.
"""
function Models.infer_dataloader(::ModernNCAModel, info, data, dev, ps=nothing, st=nothing)
    ref = info[:nca_ref]
    cx, cy = dev(ref.cx), dev(ref.cy)
    return Iterators.map(x -> (x, cx, cy), data)
end

"""
    Models.eval_dataloader(m::ModernNCAModel, ...)

Wrap each eval batch as `((x, cx, cy), rest...)` with the raw training corpus.
The corpus is encoded inside the forward pass so embeddings always use the
current model parameters rather than a stale pre-encoded version.
"""
function Models.eval_dataloader(::ModernNCAModel, info, data, dev, ps=nothing, st=nothing)
    ref = info[:nca_ref]
    cx, cy = dev(ref.cx), dev(ref.cy)
    return Iterators.map(d -> ((d[1], cx, cy), Base.tail(d)...), data)
end

end # module
