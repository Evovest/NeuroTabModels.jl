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

"""
    _pairwise_dist(q, k, ϵ)

Pairwise Euclidean distance between column vectors. `q` is `(d, b)`, `k` is
`(d, n)`; returns an `(n, b)` matrix with `ϵ` added under the square root for
numerical stability.
"""
function _pairwise_dist(q::AbstractMatrix, k::AbstractMatrix, ϵ::Float32)
    q2 = sum(abs2, q; dims=1)
    k2 = sum(abs2, k; dims=1)
    sqrt.(max.(0f0, permutedims(k2, (2, 1)) .+ q2 .- 2f0 .* (k' * q)) .+ ϵ)
end

"""
    _mask_diag(d, batch)

Mask self-pairs in the distance matrix when the minibatch has been concatenated
onto the front of the candidate set: for `i` in `1:batch`, query `i` corresponds
to candidate `i`, so `d[i, i]` is set to `typemax(T)` to suppress retrieval.
"""
function _mask_diag(d::AbstractMatrix{T}, batch::Int) where {T}
    r, c = size(d)
    ri = reshape(1:r, :, 1)
    cj = reshape(1:c, 1, :)
    ifelse.((ri .== cj) .& (ri .≤ batch), typemax(T), d)
end

"""
Shape the attention weights `α` and candidate targets `cy` into what the
standard `Losses` functions expect:

- `MSE`/`MAE`: weighted regression predictions, `(1, batch)`.
- `LogLoss`:   logits, `(1, batch)` (inverse-sigmoid of the soft probability).
- `MLogLoss`:  log-probabilities, `(outsize, batch)`. `logsoftmax` is idempotent
  on normalised log-probs, so `mlogloss` consumes them unchanged.
"""
_to_output(::Type{<:Union{MSE,MAE}}, α, cy, _) =
    reshape(α' * reshape(Float32.(cy), :, 1), 1, :)

function _to_output(::Type{<:LogLoss}, α, cy, _)
    p = reshape(α' * reshape(Float32.(cy), :, 1), 1, :)
    p = clamp.(p, 1f-6, 1f0 - 1f-6)
    log.(p ./ (1f0 .- p))
end

function _to_output(::Type{<:MLogLoss}, α, cy, outsize::Int)
    yk = reshape(cy, :)
    oh = reduce(hcat, [Float32.(yk .== c) for c in 1:outsize])
    log.(clamp.(permutedims(α' * oh, (2, 1)), 1f-7, Inf32))
end

"""
    ModernNCAModel(backbone, cfg, outsize, loss_type)

`backbone` is `Chain(embedding, encoder)`. We extend state with
`training=Val(true)` so `Lux.testmode` flips it; the forward uses that flag to
gate the self-pair augmentation + diagonal mask.
"""
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

function (cfg::ModernNCAConfig)(; nfeats, outsize,
                                 loss_type::Type{<:LossType}=MSE,
                                 embedding_layer=nothing)
    emb = isnothing(embedding_layer) ? WrappedFunction(identity) : embedding_layer
    backbone = Chain(emb, _encoder(cfg, nfeats))
    ModernNCAModel(backbone, cfg, Int(outsize), loss_type)
end

"""
    (m::ModernNCAModel)((x, cand_x, cand_y), ps, st)

Single forward signature for both modes. `Lux.StatefulLuxLayer` runs the same
backbone over the query batch and the candidate set, accumulating BatchNorm
statistics correctly. The training-mode flag in `st.training` gates the
self-pair concatenation + diagonal mask used during training.

At training time, the loader yields `cand_y = vcat(batch_y, sampled_complement_y)`
so the augmented candidate row order matches the column order of
`hcat(zq, zk)`.
"""
function (m::ModernNCAModel)(input::Tuple, ps, st)
    x, cand_x, cand_y = input
    f = Lux.StatefulLuxLayer{true}(m.backbone, ps.backbone, st.backbone)
    zq = f(x)
    zk = f(cand_x)
    zk_eff, mask_batch = _maybe_augment(st.training, zq, zk)
    d = _pairwise_dist(zq, zk_eff, m.cfg.eps) ./ max(m.cfg.temperature, m.cfg.eps)
    d = mask_batch > 0 ? _mask_diag(d, mask_batch) : d
    α = softmax(-d; dims=1)
    out = _to_output(m.loss_type, α, cand_y, m.outsize)
    return out, (backbone=f.st, training=st.training)
end

_maybe_augment(::Val{true},  zq, zk) = (hcat(zq, zk), size(zq, 2))
_maybe_augment(::Val{false}, zq, zk) = (zk, 0)

"""
    ModernNCALoader(full_x, full_y, batchsize, sample_rate, rng)

Training data iterator. Each epoch shuffles the training indices once; each
step takes a `batchsize`-window as the minibatch and uses the complement
(optionally subsampled by `sample_rate`) as candidates.

Yields `((x, cand_x, cand_y_aug), y)` where `cand_y_aug = vcat(batch_y, complement_y)`
so the forward's `hcat(zq, zk)` augmentation lines up with `cand_y_aug` without
the model having to slice or concatenate target arrays.
"""
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
    if state === nothing
        perm = randperm(l.rng, n)
        start = 1
    else
        perm, start = state
    end
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

"""
    build_corpus(df, feature_names, target_name, loss_type, scalers)

Build `(full_x, full_y)` from the training frame in the encoding the model
expects. Stored on `m.info[:nca_ref]` so the inference hook can supply it as
the candidate set.
"""
function build_corpus(df::AbstractDataFrame, feature_names, target_name,
                      loss_type::Type{<:LossType}, scalers)
    full_x = permutedims(Matrix{Float32}(select(df, collect(feature_names))))
    full_y = _encode_targets(df, target_name, loss_type, scalers)
    full_x, full_y
end

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

# ---------- Integration hooks: localized to this module ----------

"""
    Models.build_chain(cfg::ModernNCAConfig, embed_chain; ...)

ModernNCA consumes the embedding internally (re-applying it to both queries
and the candidate set), so the embedding is passed as a model kwarg rather
than wrapped in an outer `Chain`.
"""
function Models.build_chain(cfg::ModernNCAConfig, embed_chain;
        outsize, d_in, loss_type, kw...)
    cfg(; nfeats=d_in, outsize, loss_type, embedding_layer=embed_chain)
end

"""
    Models.train_dataloader(cfg::ModernNCAConfig, m, _default, df; ...)

Replaces the default per-row dataloader with a `ModernNCALoader` that yields
the `(x, cand_x, cand_y)` tuples the model consumes. Stashes the corpus on
`m.info[:nca_ref]` for the inference hook to pick up.
"""
function Models.train_dataloader(cfg::ModernNCAConfig, m::NeuroTabModel, ::Any, df;
        feature_names, target_name, loss_type, scalers, batchsize, dev, rng, kw...)
    cx, cy = build_corpus(df, feature_names, target_name, loss_type, scalers)
    m.info[:nca_ref] = (cx=cx, cy=cy)
    return ModernNCALoader(dev(cx), dev(cy), batchsize, cfg.sample_rate, rng)
end

"""
    Models.infer_dataloader(::ModernNCAModel, info, data, dev)

Wraps each batch in the user's inference iterator with the training corpus
`(cx, cy)` so the unified forward signature `(x, cand_x, cand_y)` is
satisfied at inference.
"""
function Models.infer_dataloader(::ModernNCAModel, info, data, dev)
    ref = info[:nca_ref]
    cx = dev(ref.cx)
    cy = dev(ref.cy)
    return Iterators.map(x -> (x, cx, cy), data)
end

end # module
