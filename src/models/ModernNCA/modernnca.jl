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
    sqrt.(max.(0f0, permutedims(k2, (2, 1)) .+ q2 .- 2f0 .* (k' * q)) .+ ϵ)
end

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

function _encode(m::ModernNCAModel, x, cand_x, ps, st)
    zq, st_bb = m.backbone(x, ps, st)
    zk, st_bb = m.backbone(cand_x, ps, st_bb)
    zq, zk, st_bb
end

function _nca_logits(m::ModernNCAModel, zq, zk, cand_y)
    d = _pairwise_dist(zq, zk, m.cfg.eps) ./ max(m.cfg.temperature, m.cfg.eps)
    α = softmax(-d; dims=1)
    _to_output(m.loss_type, α, cand_y, m.outsize)
end

function _nca_logits(m::ModernNCAModel, zq, zk, cand_y, dist_mask)
    d = _pairwise_dist(zq, zk, m.cfg.eps) ./ max(m.cfg.temperature, m.cfg.eps)
    α = softmax(-(d .+ dist_mask); dims=1)
    _to_output(m.loss_type, α, cand_y, m.outsize)
end

"Training forward: concat encoded queries into candidates and mask the diagonal."
function (m::ModernNCAModel)((x, cand_x, cand_y, y, dist_mask)::Tuple{Any,Any,Any,Any,Any}, ps, st)
    zq, zk, st_bb = _encode(m, x, cand_x, ps, st)
    zk = hcat(zq, zk)
    cy = vcat(vec(y), vec(cand_y))
    out = _nca_logits(m, zq, zk, cy, dist_mask)
    return out, st_bb
end

"Inference / eval forward: queries attend to the corpus only."
function (m::ModernNCAModel)((x, cand_x, cand_y)::Tuple{Any,Any,Any}, ps, st)
    zq, zk, st_bb = _encode(m, x, cand_x, ps, st)
    out = _nca_logits(m, zq, zk, cand_y)
    return out, st_bb
end

"Training iterator. Corpus is dev'd once at construction; per-batch gathers happen on-device."
struct ModernNCALoader{X,Y,R<:AbstractRNG,D,M}
    full_x::X
    full_y::Y
    batchsize::Int
    sample_rate::Float32
    rng::R
    dev::D
    train_mask::M
end

Base.length(l::ModernNCALoader) = fld(size(l.full_x, 2), l.batchsize)

function Base.iterate(l::ModernNCALoader, state=nothing)
    n = size(l.full_x, 2)
    perm, start = state === nothing ? (randperm(l.rng, n), 1) : state
    stop = start + l.batchsize - 1
    stop > n && return nothing
    batch_idx = perm[start:stop]
    cand_full = vcat(@view(perm[1:start-1]), @view(perm[stop+1:end]))
    cand_idx = if l.sample_rate < 1f0
        m = length(cand_full)
        k = Int(floor(l.sample_rate * m))
        k == 0 ? cand_full[1:0] :
            k < m ? cand_full[randperm(l.rng, m)[1:k]] : cand_full
    else
        cand_full
    end
    bidx = l.dev(batch_idx)
    cidx = l.dev(cand_idx)
    x = l.full_x[:, bidx]
    y = l.full_y[bidx]
    cand_x = l.full_x[:, cidx]
    cand_y = l.full_y[cidx]
    return ((x, cand_x, cand_y, y, l.train_mask), y), (perm, stop + 1)
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
    k = cfg.sample_rate < 1f0 ? Int(floor(cfg.sample_rate * (n - batchsize))) : n - batchsize
    mask_rows = batchsize + max(k, 0)
    train_mask = zeros(Float32, mask_rows, batchsize)
    for i in 1:batchsize
        train_mask[i, i] = Inf32
    end
    return ModernNCALoader(dev(cx), dev(cy), batchsize, cfg.sample_rate, rng, dev, dev(train_mask))
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
