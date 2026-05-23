module ModernNCA

export ModernNCAConfig

import ..Models
import ..Models: Architecture, NeuroTabModel
import ..Losses: LossType, MSE, MAE, LogLoss, MLogLoss

using Lux
using LuxCore
using NNlib: relu, softmax
using Random: randperm, AbstractRNG
using DataFrames: AbstractDataFrame, select
using CategoricalArrays

"""
    ModernNCAConfig(; d_embedding=128, n_blocks=2, d_block=256,
                     dropout=0.1, temperature=1.0, sample_rate=0.8, eps=1f-8)

Hyperparameters for a ModernNCA architecture. Pass to `NeuroTabRegressor` or
`NeuroTabClassifier` as the `arch` argument.

# Arguments
- `d_embedding`: dimension of the encoder output.
- `n_blocks`, `d_block`: count and hidden width of the post-encoder MLP blocks
  after the linear encoder. `n_blocks=0` keeps only the linear encoder.
- `dropout`: dropout rate inside each MLP block. Skipped when `<= 0`.
- `temperature`: softmax temperature on the negative distances.
- `sample_rate`: fraction of the minibatch complement kept as candidates each
  training step (Stochastic Neighborhood Sampling). `1.0` disables sampling.
- `eps`: numerical floor inside `sqrt` for the pairwise distance and as a
  denominator floor on `temperature`.
"""
struct ModernNCAConfig <: Architecture
    d_embedding::Int
    n_blocks::Int
    d_block::Int
    dropout::Float32
    temperature::Float32
    sample_rate::Float32
    eps::Float32
end
function ModernNCAConfig(;
    d_embedding::Int=128, n_blocks::Int=2, d_block::Int=256,
    dropout::Real=0.1, temperature::Real=1.0, sample_rate::Real=0.8, eps::Real=1f-8,
)
    ModernNCAConfig(d_embedding, n_blocks, d_block,
        Float32(dropout), Float32(temperature), Float32(sample_rate), Float32(eps))
end

"Embedding -> linear encoder -> optional post-encoder MLP tower."
function _backbone(cfg::ModernNCAConfig, d_in::Int, embedding_layer)
    emb = isnothing(embedding_layer) ? WrappedFunction(identity) : embedding_layer
    layers = Any[emb, Dense(d_in => cfg.d_embedding)]
    for _ in 1:cfg.n_blocks
        push!(layers, BatchNorm(cfg.d_embedding))
        push!(layers, Dense(cfg.d_embedding => cfg.d_block, relu))
        cfg.dropout > 0 && push!(layers, Dropout(cfg.dropout))
        push!(layers, Dense(cfg.d_block => cfg.d_embedding))
    end
    cfg.n_blocks > 0 && push!(layers, BatchNorm(cfg.d_embedding))
    Chain(layers...)
end

function _pairwise_dist(q::AbstractMatrix, k::AbstractMatrix, ϵ::Float32)
    q2 = sum(abs2, q; dims=1)
    k2 = sum(abs2, k; dims=1)
    kq = k' * q
    k2c = reshape(k2, :, 1)
    @. sqrt(max(0f0, k2c + q2 - 2f0 * kq) + ϵ)
end

_diag_inf(i, j, d) = ifelse(i == j, typemax(typeof(d)), d)
_mask_diag(d::AbstractMatrix) =
    _diag_inf.(reshape(1:size(d, 1), :, 1), reshape(1:size(d, 2), 1, :), d)

_to_output(::Type{<:Union{MSE,MAE}}, α, cy, _) = reshape(α' * cy, 1, :)

function _to_output(::Type{<:LogLoss}, α, cy, _)
    p = clamp.(reshape(α' * cy, 1, :), 1f-6, 1f0 - 1f-6)
    log.(p ./ (1f0 .- p))
end

function _to_output(::Type{<:MLogLoss}, α, cy, outsize::Int)
    oh = ((k, c) -> ifelse(k == c, 1f0, 0f0)).(
        reshape(UInt32(1):UInt32(outsize), :, 1), reshape(cy, 1, :))
    log.(clamp.(oh * α, 1f-7, Inf32))
end

struct ModernNCAModel{B,LT} <: LuxCore.AbstractLuxWrapperLayer{:backbone}
    backbone::B
    cfg::ModernNCAConfig
    outsize::Int
    loss_type::Type{LT}
end

"Construct a `ModernNCAModel` from the config, sizing, and optional embedding."
function (cfg::ModernNCAConfig)(; nfeats, outsize,
                                 loss_type::Type{<:LossType}=MSE,
                                 embedding_layer=nothing)
    ModernNCAModel(_backbone(cfg, nfeats, embedding_layer), cfg, Int(outsize), loss_type)
end

function _nca_logits(m::ModernNCAModel, zq, zk, cy; mask_self::Bool=false)
    d = _pairwise_dist(zq, zk, m.cfg.eps) ./ max(m.cfg.temperature, m.cfg.eps)
    mask_self && (d = _mask_diag(d))
    α = softmax(-d; dims=1)
    _to_output(m.loss_type, α, cy, m.outsize)
end

# Chunked corpus encode to bound peak memory and stay within
# backend-specific shape limits on the BatchNorm forward pass.
function _encode_corpus(m::ModernNCAModel, cx, ps, st; chunk::Int=2048)
    n = size(cx, 2)
    n <= chunk && return m.backbone(cx, ps, st)[1]
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

"Training forward: queries and candidates are encoded in one backbone pass so
they share BatchNorm batch statistics; self-distances on the diagonal are
masked inline so queries don't attend to themselves."
function (m::ModernNCAModel)((x, cand_x, cand_y, y)::Tuple{Any,Any,Any,Any}, ps, st)
    B = size(x, 2)
    z_all, st_bb = m.backbone(hcat(x, cand_x), ps, st)
    zq = z_all[:, 1:B]
    cy = vcat(vec(y), vec(cand_y))
    return _nca_logits(m, zq, z_all, cy; mask_self=true), st_bb
end

"Inference / eval forward: queries attend to the corpus only; corpus is
encoded in chunks to keep peak memory bounded."
function (m::ModernNCAModel)((x, cand_x, cand_y)::Tuple{Any,Any,Any}, ps, st)
    zq, st_bb = m.backbone(x, ps, st)
    zk = _encode_corpus(m, cand_x, ps, st_bb)
    return _nca_logits(m, zq, zk, cand_y), st_bb
end

"Training iterator. Corpus is moved to device once at construction; each
step samples `n_cand` candidate indices from outside the batch window."
struct ModernNCALoader{X,Y,R<:AbstractRNG,D}
    full_x::X
    full_y::Y
    batchsize::Int
    n_cand::Int
    rng::R
    dev::D
end

Base.length(l::ModernNCALoader) = fld(size(l.full_x, 2), l.batchsize)

function Base.iterate(l::ModernNCALoader, state=nothing)
    n = size(l.full_x, 2)
    perm, start = state === nothing ? (randperm(l.rng, n), 1) : state
    stop = start + l.batchsize - 1
    stop > n && return nothing

    batch_idx = perm[start:stop]
    cand_idx = Vector{Int}(undef, l.n_cand)
    m = n - l.batchsize
    @inbounds for i in 1:l.n_cand
        j = rand(l.rng, 1:m)
        cand_idx[i] = j < start ? perm[j] : perm[j + l.batchsize]
    end

    bidx = l.dev(batch_idx)
    cidx = l.dev(cand_idx)
    x = l.full_x[:, bidx]
    y = l.full_y[bidx]
    cand_x = l.full_x[:, cidx]
    cand_y = l.full_y[cidx]
    return ((x, cand_x, cand_y, y), y), (perm, stop + 1)
end

"Build the training corpus `(full_x, full_y)` in the encoding the model expects."
function build_corpus(df::AbstractDataFrame, feature_names, target_name,
                      loss_type::Type{<:LossType}, scalers)
    full_x = permutedims(Matrix{Float32}(select(df, collect(feature_names))))
    full_y = _encode_targets(df, target_name, loss_type, scalers)
    full_x, full_y
end

"Encode target column according to the loss type."
_encode_targets(df, target_name, ::Type{<:MLogLoss}, _) =
    UInt32.(CategoricalArrays.levelcode.(df[!, target_name]))

function _encode_targets(df, target_name, ::Type{<:LogLoss}, _)
    col = df[!, target_name]
    eltype(col) <: CategoricalValue || return Float32.(col)
    levels = CategoricalArrays.levels(col)
    length(levels) == 2 || error("For `loss=:logloss`, target must have exactly 2 classes.")
    Float32.(CategoricalArrays.levelcode.(col) .- 1)
end

function _encode_targets(df, target_name, ::Type{<:Union{MSE,MAE}}, scalers)
    y = Float32.(df[!, target_name])
    isnothing(scalers) || (y .= (y .- scalers.mu) ./ scalers.sigma)
    y
end

"Pass the embedding into the model rather than wrapping it in an outer `Chain`."
Models.build_chain(cfg::ModernNCAConfig, embed_chain;
        outsize, d_in, loss_type, kw...) =
    cfg(; nfeats=d_in, outsize, loss_type, embedding_layer=embed_chain)

"Return the candidate-set loader and stash the corpus on `m.info[:nca_ref]`."
function Models.train_dataloader(cfg::ModernNCAConfig, m::NeuroTabModel, ::Any, df;
        feature_names, target_name, loss_type, scalers, batchsize, dev, rng, kw...)
    cx, cy = build_corpus(df, feature_names, target_name, loss_type, scalers)
    m.info[:nca_ref] = (cx=cx, cy=cy)
    n = size(cx, 2)
    n_cand = cfg.sample_rate >= 1f0 ? max(n - batchsize, 1) :
        max(Int(floor(cfg.sample_rate * (n - batchsize))), 1)
    return ModernNCALoader(dev(cx), dev(cy), batchsize, n_cand, rng, dev)
end

"Wrap each inference batch with the training corpus to match the forward signature."
function Models.infer_dataloader(::ModernNCAModel, info, data, dev)
    ref = info[:nca_ref]
    cx, cy = dev(ref.cx), dev(ref.cy)
    return Iterators.map(x -> (x, cx, cy), data)
end

"Wrap each eval batch's `x` with the training corpus; leave `(y, w, offset)` untouched."
function Models.eval_dataloader(::ModernNCAModel, info, data, dev)
    ref = info[:nca_ref]
    cx, cy = dev(ref.cx), dev(ref.cy)
    return Iterators.map(d -> ((d[1], cx, cy), Base.tail(d)...), data)
end

end # module
