module ModernNCA

export ModernNCAConfig

import ..Models
import ..Models: Architecture, NeuroTabModel
import ..Losses: LossType, MSE, MAE, LogLoss, MLogLoss

using Lux
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
- `n_blocks`, `d_block`: count and hidden width of the encoder's residual MLP
  blocks. `n_blocks=0` collapses the encoder to a single `Dense` projection.
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

"Build the residual MLP encoder."
function _encoder(cfg::ModernNCAConfig, d_in::Int)
    layers = Any[Dense(d_in => cfg.d_embedding)]
    for _ in 1:cfg.n_blocks
        push!(layers, BatchNorm(cfg.d_embedding))
        push!(layers, Dense(cfg.d_embedding => cfg.d_block))
        push!(layers, WrappedFunction(relu))
        cfg.dropout > 0 && push!(layers, Dropout(cfg.dropout))
        push!(layers, Dense(cfg.d_block => cfg.d_embedding))
    end
    cfg.n_blocks > 0 && push!(layers, BatchNorm(cfg.d_embedding))
    Chain(layers...)
end

"Pairwise Euclidean distance between columns of `q` and `k`, with `ϵ` added under the square root."
function _pairwise_dist(q::AbstractMatrix, k::AbstractMatrix, ϵ::Float32)
    q2 = sum(abs2, q; dims=1)
    k2 = sum(abs2, k; dims=1)
    sqrt.(max.(0f0, permutedims(k2, (2, 1)) .+ q2 .- 2f0 .* (k' * q)) .+ ϵ)
end

"""
Set `d[i, i] = typemax(T)` for `i ≤ batch` to suppress self-pair retrieval.
Fused into one broadcast to avoid materialising a CPU `BitMatrix`, which is
not bitstype and would fail on GPU.
"""
function _mask_diag(d::AbstractMatrix{T}, batch::Int) where {T}
    r, c = size(d)
    inf = typemax(T)
    ((i, j, x) -> ifelse((i == j) & (i <= batch), inf, x)).(
        reshape(1:r, :, 1), reshape(1:c, 1, :), d)
end

"Reshape attention weights and candidate targets into the shape required by each loss."
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

"ModernNCA model container. State carries `training` for `Lux.testmode` to toggle."
struct ModernNCAModel{B,LT} <: Lux.AbstractLuxContainerLayer{(:backbone,)}
    backbone::B
    cfg::ModernNCAConfig
    outsize::Int
    loss_type::Type{LT}
end

Lux.initialstates(rng::AbstractRNG, m::ModernNCAModel) = (
    backbone = Lux.initialstates(rng, m.backbone),
    training = Val(true),
)

"Construct a `ModernNCAModel` from the config, sizing, and optional embedding."
function (cfg::ModernNCAConfig)(; nfeats, outsize,
                                 loss_type::Type{<:LossType}=MSE,
                                 embedding_layer=nothing)
    emb = isnothing(embedding_layer) ? WrappedFunction(identity) : embedding_layer
    backbone = Chain(emb, _encoder(cfg, nfeats))
    ModernNCAModel(backbone, cfg, Int(outsize), loss_type)
end

"Compute scaled pairwise distances. Training mode adds the self-pair concat and diagonal mask."
function _train_distances(::Val{true}, zq, zk, cfg)
    d = _pairwise_dist(zq, hcat(zq, zk), cfg.eps) ./ max(cfg.temperature, cfg.eps)
    _mask_diag(d, size(zq, 2))
end
_train_distances(::Val{false}, zq, zk, cfg) =
    _pairwise_dist(zq, zk, cfg.eps) ./ max(cfg.temperature, cfg.eps)

"Forward pass. The loader prefixes `cand_y` with the batch targets so it lines up with `hcat(zq, zk)`."
function (m::ModernNCAModel)((x, cand_x, cand_y)::Tuple, ps, st)
    f = Lux.StatefulLuxLayer{true}(m.backbone, ps.backbone, st.backbone)
    zq, zk = f(x), f(cand_x)
    d = _train_distances(st.training, zq, zk, m.cfg)
    α = softmax(-d; dims=1)
    out = _to_output(m.loss_type, α, cand_y, m.outsize)
    return out, (backbone=f.st, training=st.training)
end

"Training iterator. Each step yields a minibatch and its complement (optionally subsampled) as candidates."
struct ModernNCALoader{X,Y,R<:AbstractRNG}
    full_x::X
    full_y::Y
    batchsize::Int
    sample_rate::Float32
    rng::R
end

Base.length(l::ModernNCALoader) = cld(size(l.full_x, 2), l.batchsize)

function Base.iterate(l::ModernNCALoader, state=nothing)
    n = size(l.full_x, 2)
    perm, start = state === nothing ? (randperm(l.rng, n), 1) : state
    start > n && return nothing
    stop = min(start + l.batchsize - 1, n)
    batch_idx = perm[start:stop]
    cand_full = vcat(@view(perm[1:start-1]), @view(perm[stop+1:end]))
    cand_idx = if l.sample_rate < 1f0
        m = length(cand_full)
        k = max(1, floor(Int, l.sample_rate * m))
        k < m ? cand_full[randperm(l.rng, m)[1:k]] : cand_full
    else
        cand_full
    end
    x = l.full_x[:, batch_idx]
    y = l.full_y[batch_idx]
    cand_x = l.full_x[:, cand_idx]
    cand_y = vcat(y, l.full_y[cand_idx])
    return ((x, cand_x, cand_y), y), (perm, stop + 1)
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
    return ModernNCALoader(dev(cx), dev(cy), batchsize, cfg.sample_rate, rng)
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
