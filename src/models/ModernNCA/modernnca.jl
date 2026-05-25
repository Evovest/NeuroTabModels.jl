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
                     dropout=0.1, temperature=1.0, sample_rate=0.8, eps=1f-8)

Hyperparameters for ModernNCA. Pass as the `arch` argument to `NeuroTabRegressor`
or `NeuroTabClassifier`.

# Arguments
- `d_embedding`: encoder output dimension.
- `n_blocks`, `d_block`: number and hidden width of post-encoder MLP blocks.
- `dropout`: dropout rate inside each block; skipped when `<= 0`.
- `temperature`: softmax temperature on negative pairwise distances.
- `sample_rate`: fraction of the batch complement sampled as candidates per step
  (Stochastic Neighborhood Sampling). `1.0` uses the full complement.
- `eps`: numerical floor in `sqrt` and as a `temperature` lower bound.
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
moved to device once at construction; each step draws an independent random batch.

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
    d_embedding::Int=128, n_blocks::Int=2, d_block::Int=256,
    dropout::Real=0.1, temperature::Real=1.0, sample_rate::Real=0.8, eps::Real=1f-8,
)
    return ModernNCAConfig(d_embedding, n_blocks, d_block,
        Float32(dropout), Float32(temperature), Float32(sample_rate), Float32(eps))
end

"""
    _backbone(cfg, d_in, embedding_layer)

Build the ModernNCA encoder: embedding → linear → n_blocks × (BN → Dense(relu) → Dropout → Dense) → BN.
"""
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
    return Chain(layers...)
end

function (cfg::ModernNCAConfig)(; nfeats, outsize,
                                 loss_type::Type{<:LossType}=MSE,
                                 embedding_layer=nothing)
    return ModernNCAModel(_backbone(cfg, nfeats, embedding_layer), cfg, Int(outsize), loss_type)
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
    return sqrt.(max.(0f0, k2' .+ q2 .- 2f0 .* kq) .+ ϵ)
end

_diag_inf(i, j, d) = ifelse(i == j, typemax(typeof(d)), d)

"""
    _mask_diag(d) -> Matrix

Set the diagonal of distance matrix `d` to `typemax`, collapsing self-attention
weights to zero during training.
"""
_mask_diag(d::AbstractMatrix) =
    _diag_inf.(reshape(1:size(d, 1), :, 1), reshape(1:size(d, 2), 1, :), d)

"""
    _finalize_output(loss_type, out) -> Matrix

Apply the loss-specific transform to a pre-computed weighted-mean matrix `out`.
- `MSE/MAE/GaussianMLE`: identity (regression output).
- `LogLoss`: log-odds of the weighted probability.
- `MLogLoss`: log of weighted class probability vector.
"""
_finalize_output(::Type{<:Union{MSE,MAE,GaussianMLE}}, out) = out

function _finalize_output(::Type{<:LogLoss}, out)
    p = clamp.(out, 1f-6, 1f0 - 1f-6)
    return log.(p ./ (1f0 .- p))
end

_finalize_output(::Type{<:MLogLoss}, out) = log.(clamp.(out, 1f-7, Inf32))

"""
    _to_output(loss_type, α, cy, outsize) -> Matrix

Compute model output from softmax weights `α` `(N, batch)` and targets `cy` `(N,)`.
Builds the weighted sum then delegates to `_finalize_output` for the final transform.
"""
function _to_output(::Type{LT}, α, cy, outsize::Int) where {LT<:LossType}
    if LT <: MLogLoss
        oh = ((k, c) -> ifelse(k == c, 1f0, 0f0)).(
            reshape(UInt32(1):UInt32(outsize), :, 1), reshape(cy, 1, :))
        return _finalize_output(LT, oh * α)
    end
    return _finalize_output(LT, reshape(α' * cy, 1, :))
end

"""
    _chunk_numerator(m, weights, chunk_y) -> Matrix

Softmax numerator contribution for one corpus chunk.
Returns `reshape(chunk_y, 1, :) * weights` for scalar targets, or the
one-hot weighted sum `(outsize, batch)` for `MLogLoss`.
"""
function _chunk_numerator(m::ModernNCAModel, weights, chunk_y)
    if m.loss_type <: MLogLoss
        oh = ((k, c) -> ifelse(k == c, 1f0, 0f0)).(
            reshape(UInt32(1):UInt32(m.outsize), :, 1), reshape(chunk_y, 1, :))
        return oh * weights
    end
    return reshape(chunk_y, 1, :) * weights
end

"""
    _accumulate_chunk(m, zq, chunk_zk, chunk_y, t, num, denom, running_max)

Update online-softmax accumulators `(num, denom, running_max)` with one corpus
chunk, using the running-max shift to keep exponentiation numerically stable
across chunks (same log-sum-exp trick as FlashAttention).

# Arguments
- `m`: ModernNCA model.
- `zq`: query embeddings.
- `chunk_zk`: key embeddings for one corpus chunk.
- `chunk_y`: targets for the chunk.
- `t`: temperature.
- `num`, `denom`, `running_max`: online-softmax accumulators.
"""
function _accumulate_chunk(m::ModernNCAModel, zq, chunk_zk, chunk_y, t, num, denom, running_max)
    scores = -_pairwise_dist(zq, chunk_zk, m.cfg.eps) ./ t
    chunk_max = maximum(scores; dims=1)
    next_max = max.(running_max, chunk_max)
    old_scale = exp.(running_max .- next_max)
    weights = exp.(scores .- next_max)
    num = num .* old_scale .+ _chunk_numerator(m, weights, chunk_y)
    denom = denom .* old_scale .+ sum(weights; dims=1)
    return num, denom, next_max
end

"""
    _streaming_nca_logits(m, zq, zk, cy; corpus_chunk=4096) -> Matrix

Large-corpus inference path. Streams the corpus in chunks of `corpus_chunk` rows,
accumulating online-softmax statistics to avoid materialising the full
`(N, batch)` distance matrix.

# Arguments
- `m`: ModernNCA model.
- `zq`: query embeddings.
- `zk`: corpus embeddings.
- `cy`: corpus targets.
- `corpus_chunk`: corpus rows per chunk.
"""
function _streaming_nca_logits(m::ModernNCAModel, zq, zk, cy; corpus_chunk::Int=4096)
    n = size(zk, 2)
    b = size(zq, 2)
    t = max(m.cfg.temperature, m.cfg.eps)

    running_max = fill!(similar(zq, 1, b), -Inf32)
    denom = fill!(similar(zq, 1, b), 0f0)
    num = fill!(similar(zq, m.loss_type <: MLogLoss ? m.outsize : 1, b), 0f0)

    n_full = fld(n, corpus_chunk)
    for i in 1:n_full
        s = (i - 1) * corpus_chunk + 1
        num, denom, running_max = _accumulate_chunk(
            m, zq, zk[:, s:s+corpus_chunk-1], cy[s:s+corpus_chunk-1], t, num, denom, running_max)
    end
    s = n_full * corpus_chunk + 1
    if s <= n
        num, denom, _ = _accumulate_chunk(m, zq, zk[:, s:n], cy[s:n], t, num, denom, running_max)
    end

    return _finalize_output(m.loss_type, num ./ denom)
end

"""
    _nca_logits(m, zq, zk, cy; mask_self=false, corpus_chunk=4096) -> Matrix

Dispatch between the dense and streaming attention paths.

Dense path is used when `mask_self=true` (training) or `N <= corpus_chunk`. The
streaming path does not support diagonal masking, so training always takes the
dense path; `N` is bounded by `batchsize + n_cand` during training, so this
does not materialise a large matrix. When `N > corpus_chunk` at inference,
the streaming path avoids materialising the full `(N, batch)` distance matrix.
"""
function _nca_logits(m::ModernNCAModel, zq, zk, cy; mask_self::Bool=false, corpus_chunk::Int=4096)
    if mask_self || size(zk, 2) <= corpus_chunk
        d = _pairwise_dist(zq, zk, m.cfg.eps) ./ max(m.cfg.temperature, m.cfg.eps)
        mask_self && (d = _mask_diag(d))
        α = softmax(-d; dims=1)
        return _to_output(m.loss_type, α, cy, m.outsize)
    end
    return _streaming_nca_logits(m, zq, zk, cy; corpus_chunk)
end

"""
    _encode_corpus(m, cx, ps, st; chunk=2048) -> Matrix

Encode raw corpus `cx` `(d_in, N)` in chunks of `chunk` rows to bound peak
memory and stay within BatchNorm shape limits. Returns `zk` `(d_embedding, N)`.
"""
function _encode_corpus(m::ModernNCAModel, cx, ps, st; chunk::Int=2048)
    n = size(cx, 2)
    n <= chunk && return m.backbone(cx, ps, st)[1]

    z0, _ = m.backbone(cx[:, 1:chunk], ps, st)
    zk = similar(z0, size(z0, 1), n)
    zk[:, 1:chunk] .= z0

    n_full = fld(n, chunk)
    for i in 2:n_full
        s = (i - 1) * chunk + 1
        zk[:, s:s+chunk-1] .= m.backbone(cx[:, s:s+chunk-1], ps, st)[1]
    end
    s = n_full * chunk + 1
    if s <= n
        zk[:, s:n] .= m.backbone(cx[:, s:n], ps, st)[1]
    end
    return zk
end


"""
    (m::ModernNCAModel)((x, cand_x, cand_y, y), ps, st)

Training forward: encodes queries and candidates in one backbone call (shared
BatchNorm statistics), then computes NCA logits with diagonal self-masking.
"""
function (m::ModernNCAModel)((x, cand_x, cand_y, y)::Tuple{Any,Any,Any,Any}, ps, st)
    B = size(x, 2)
    z_all, st_bb = m.backbone(hcat(x, cand_x), ps, st)
    cy = vcat(vec(y), vec(cand_y))
    return _nca_logits(m, z_all[:, 1:B], z_all, cy; mask_self=true), st_bb
end

"""
    (m::ModernNCAModel)((x, cx, cand_y), ps, st)

Inference/eval forward: encodes queries and the raw corpus `cx` with the current
parameters, then attends over the corpus. Encoding is done here (not pre-computed)
so the corpus embeddings always match the current model during evaluation.
"""
function (m::ModernNCAModel)((x, cx, cand_y)::Tuple{Any,Any,Any}, ps, st)
    zq, st_bb = m.backbone(x, ps, st)
    zk = _encode_corpus(m, cx, ps, st_bb)
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

    bidx = l.dev(perm[start:stop])
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

Return `(full_x, full_y)`: the corpus feature matrix `(d_in, N)` as `Float32`
and the encoded target vector, ready for `ModernNCALoader`.
"""
function build_corpus(df::AbstractDataFrame, feature_names, target_name,
                      loss_type::Type{<:LossType}, scalers)
    full_x = permutedims(Matrix{Float32}(select(df, collect(feature_names))))
    return full_x, _encode_targets(df, target_name, loss_type, scalers)
end

"""
    _encode_targets(df, target_name, loss_type, scalers)

Encode targets for each loss type:
- `MLogLoss`: 1-based `UInt32` class codes.
- `LogLoss`: `Float32` in `{0, 1}`.
- `MSE / MAE / GaussianMLE`: `Float32`, standardised when `scalers` is provided.

# Arguments
- `df`: source data frame.
- `target_name`: target column.
- `loss_type`: target encoding dispatch.
- `scalers`: optional target scaler `(mu, sigma)`.
"""
_encode_targets(df, target_name, ::Type{<:MLogLoss}, _) =
    UInt32.(CategoricalArrays.levelcode.(df[!, target_name]))

function _encode_targets(df, target_name, ::Type{<:LogLoss}, _)
    col = df[!, target_name]
    eltype(col) <: CategoricalValue || return Float32.(col)
    length(CategoricalArrays.levels(col)) == 2 ||
        error("LogLoss target must have exactly 2 classes.")
    return Float32.(CategoricalArrays.levelcode.(col) .- 1)
end

function _encode_targets(df, target_name, ::Type{<:Union{MSE,MAE,GaussianMLE}}, scalers)
    y = Float32.(df[!, target_name])
    isnothing(scalers) || (y .= (y .- scalers.mu) ./ scalers.sigma)
    return y
end

"""
    Models.build_chain(cfg::ModernNCAConfig, embed_chain; outsize, d_in, loss_type, kw...)

Pass the embedding layer into the backbone instead of wrapping it in an outer `Chain`.
"""
Models.build_chain(cfg::ModernNCAConfig, embed_chain;
        outsize, d_in, loss_type, kw...) =
    cfg(; nfeats=d_in, outsize, loss_type, embedding_layer=embed_chain)

"""
    Models.train_dataloader(cfg::ModernNCAConfig, ...)

Build `ModernNCALoader` and stash the raw corpus on `m.info[:nca_ref]`.
`n_cand = floor(sample_rate × (N − batchsize))`; `sample_rate ≥ 1` uses the full
complement; `batchsize == N` gives `n_cand = 0`.

# Arguments
- `cfg`: ModernNCA config.
- `m`: fitted model wrapper.
- `df`: training data frame.
- `feature_names`, `target_name`: data columns.
- `loss_type`, `scalers`: target encoding metadata.
- `batchsize`, `dev`, `rng`: loader settings.
"""
function Models.train_dataloader(cfg::ModernNCAConfig, m::NeuroTabModel, ::Any, df;
        feature_names, target_name, loss_type, scalers, batchsize, dev, rng, kw...)
    cx, cy = build_corpus(df, feature_names, target_name, loss_type, scalers)
    m.info[:nca_ref] = (cx=cx, cy=cy)
    n = size(cx, 2)
    batchsize = min(batchsize, n)
    pool = n - batchsize
    n_cand = pool == 0 ? 0 :
        cfg.sample_rate >= 1f0 ? pool :
        max(Int(floor(cfg.sample_rate * pool)), 1)
    return ModernNCALoader(dev(cx), dev(cy), batchsize, n_cand, rng, dev)
end

"""
    Models.infer_dataloader(m::ModernNCAModel, ...)

Wrap each inference batch as `(x, cx, cy)` with the raw training corpus.
The corpus is encoded inside the forward pass using the current parameters.
"""
function Models.infer_dataloader(m::ModernNCAModel, info, data, dev, _, _)
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
function Models.eval_dataloader(m::ModernNCAModel, info, data, dev, _, _)
    ref = info[:nca_ref]
    cx, cy = dev(ref.cx), dev(ref.cy)
    return Iterators.map(d -> ((d[1], cx, cy), Base.tail(d)...), data)
end

end # module
