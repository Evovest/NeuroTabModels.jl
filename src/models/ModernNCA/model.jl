"""
    ModernNCAModel

Lux container holding the feature embedding and the backbone encoder. The
embedding lives inside the model (rather than in front of it as for other
architectures) because reference rows must go through the same pipeline as
queries.
"""
struct ModernNCAModel{E,B,LT} <: LuxCore.AbstractLuxContainerLayer{(:embedding, :backbone)}
    embedding::E
    backbone::B
    cfg::ModernNCAConfig
    outsize::Int
    loss_type::Type{LT}
end

"""
    _backbone(cfg, ins)

Linear, then `n_blocks` × (BN, Dense(relu), Dropout, Dense), then BN.
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

function _build_model(cfg, embedding, ins, outsize, loss_type)
    loss_type <: Union{MSE,MAE,LogLoss,MLogLoss} ||
        throw(ArgumentError("ModernNCA does not support $(nameof(loss_type))"))
    return ModernNCAModel(embedding, _backbone(cfg, ins), cfg, Int(outsize), loss_type)
end

_temperature(m::ModernNCAModel) = max(m.cfg.temperature, m.cfg.eps)
_chunks(n::Int, chunk::Int) = Iterators.partition(1:n, chunk)

"""
    _pairwise_dist(q, k, ϵ) -> Matrix

`(num_keys, batch)` Euclidean distances between `q` `(d, batch)` and `k`
`(d, num_keys)`, via ‖q‖² + ‖k‖² − 2qᵀk so the only heavy operation is one GEMM.
"""
function _pairwise_dist(q::AbstractMatrix, k::AbstractMatrix, ϵ::Float32)
    q2 = sum(abs2, q; dims=1)
    k2 = sum(abs2, k; dims=1)
    return sqrt.(max.(0.0f0, k2' .+ q2 .- 2.0f0 .* (k' * q)) .+ ϵ)
end

_diag_inf(i, j, d) = ifelse(i == j, typemax(typeof(d)), d)
_mask_diag(d::AbstractMatrix) =
    _diag_inf.(reshape(1:size(d, 1), :, 1), reshape(1:size(d, 2), 1, :), d)

"""
    _scores(m, q, k; mask_self=false) -> (d, s)

Distances `d` and scores `s = -d / temperature`, both `(num_keys, batch)`.
`mask_self` sends the diagonal to `-Inf` so a query cannot attend to itself
(training, where the batch is part of the key set). `d` is returned unmasked
because the backward needs it.
"""
function _scores(m::ModernNCAModel, q, k; mask_self::Bool=false)
    d = _pairwise_dist(q, k, m.cfg.eps)
    return d, -(mask_self ? _mask_diag(d) : d) ./ _temperature(m)
end

"""
    _target_layout(loss_type, y)

Store targets per loss: a `(1, N)` row for regression/binary, class codes for
multiclass (a `(K, N)` one-hot is only built per chunk).
"""
_target_layout(::Type{<:Union{MSE,MAE,LogLoss}}, y) = reshape(y, 1, :)
_target_layout(::Type{<:MLogLoss}, y) = y

"""
    _target_block(loss_type, y, idx, outsize)

`(outsize, length(idx))` target block for keys `idx`.
"""
_target_block(::Type{<:Union{MSE,MAE,LogLoss}}, y, idx, _) = view(y, :, idx)
_target_block(::Type{<:MLogLoss}, y, idx, n::Int) =
    ((k, c) -> ifelse(k == c, 1.0f0, 0.0f0)).(
        reshape(UInt32(1):UInt32(n), :, 1), reshape(view(y, idx), 1, :))

_target_block(m::ModernNCAModel, y, idx) = _target_block(m.loss_type, y, idx, m.outsize)

"""
    _finalize(loss_type, p)

Map the weighted target average to what each loss expects: a mean (MSE), a
probability (LogLoss), or a class distribution (MLogLoss).
"""
_finalize(::Type{<:Union{MSE,MAE}}, p) = p
_finalize(::Type{<:LogLoss}, p) = (p = clamp.(p, 1.0f-6, 1.0f0 - 1.0f-6); log.(p ./ (1.0f0 .- p)))
_finalize(::Type{<:MLogLoss}, p) = log.(clamp.(p, 1.0f-7, Inf32))

"""
    _softmax_acc(zq, outsize) -> (running_max, denominator, numerator)

Accumulator for a softmax-weighted target average, each `(·, batch)`, folded
one key block at a time.
"""
function _softmax_acc(zq, outsize::Int)
    B = size(zq, 2)
    return (fill!(similar(zq, 1, B), -Inf32), fill!(similar(zq, 1, B), 0.0f0),
        fill!(similar(zq, outsize, B), 0.0f0))
end

"""
    _softmax_fold(acc, s, yk)

Fold score block `s` `(num_keys, batch)` with key targets `yk`
`(outsize, num_keys)`. Rescales the accumulator whenever the running max moves,
so the result equals a dense softmax while only this block is live.
"""
function _softmax_fold(acc, s, yk)
    running_max, denominator, numerator = acc
    block_max = maximum(s; dims=1)
    w = exp.(s .- block_max)
    merged = max.(running_max, block_max)
    old_scale, new_scale = exp.(running_max .- merged), exp.(block_max .- merged)
    return (merged, denominator .* old_scale .+ sum(w; dims=1) .* new_scale,
        numerator .* old_scale .+ (yk * w) .* new_scale)
end

"""
    _softmax_result(acc) -> (p, lse)

Weighted target average and per-query log-sum-exp.
"""
function _softmax_result(acc)
    running_max, denominator, numerator = acc
    return numerator ./ denominator, running_max .+ log.(denominator)
end

"""
    _encode(m, x, ps, st) -> (z, st)

Embedding then backbone. Returns `z` `(d_embedding, batch)` and the new state.
"""
function _encode(m::ModernNCAModel, x, ps, st)
    x, st_embedding = m.embedding(x, ps.embedding, st.embedding)
    z, st_backbone = m.backbone(x, ps.backbone, st.backbone)
    return z, (embedding=st_embedding, backbone=st_backbone)
end

"""
    _encode_all(m, x, ps, st) -> Matrix

Encode `x` `(ins, N)` chunk by chunk into a preallocated `(d_embedding, N)`.
"""
function _encode_all(m::ModernNCAModel, x, ps, st)
    n = size(x, 2)
    z = similar(x, m.cfg.d_embedding, n)
    for idx in _chunks(n, m.cfg.corpus_chunk_size)
        z[:, idx] .= first(_encode(m, x[:, idx], ps, st))
    end
    return z
end

"""
    Corpus(x, y, info)

The full reference set used as keys outside training: raw features `x`
`(ins, N)` on device, targets `y` in loss layout, and a cached encoding `z`.
`z` is (re)computed when `info[:nrounds]` differs from `encoded_at`: during
`fit`, parameters change only in `fit_iter!`, which bumps `nrounds`, so the
eval callback re-encodes once per round; during `infer`, `nrounds` is fixed,
so the first batch encodes and every later batch reuses it. Marked as a
Functors leaf so device moves of `(x, corpus)` batches leave the resident
corpus alone.
"""
mutable struct Corpus{X,Y,I}
    x::X
    y::Y
    info::I
    z::Any
    encoded_at::Int
end

Corpus(x, y, info) = Corpus(x, y, info, nothing, -1)

Functors.@leaf Corpus

function _keys(m::ModernNCAModel, corpus::Corpus, ps, st)
    round = corpus.info[:nrounds]
    if corpus.z === nothing || corpus.encoded_at != round
        corpus.z = _encode_all(m, corpus.x, ps, st)
        corpus.encoded_at = round
    end
    return corpus.z
end

"""
    (m::ModernNCAModel)((x, corpus::Corpus), ps, st)

Eval / inference forward: attend from the encoded query batch over the full
corpus, one chunk at a time.
"""
function (m::ModernNCAModel)((x, corpus)::Tuple{Any,Corpus}, ps, st)
    zq, st = _encode(m, x, ps, st)
    zk = _keys(m, corpus, ps, st)
    acc = _softmax_acc(zq, m.outsize)
    for idx in _chunks(size(zk, 2), m.cfg.corpus_chunk_size)
        _, s = @views _scores(m, zq, zk[:, idx])
        acc = _softmax_fold(acc, s, _target_block(m, corpus.y, idx))
    end
    p, _ = _softmax_result(acc)
    return _finalize(m.loss_type, p), st
end

"""
    (m::ModernNCAModel)((x, cand_x, cand_y, y), ps, st)

Training forward: the query batch attends over `[itself (self-masked); candidates]`.
Candidates are encoded and attended per chunk inside [`_attend_train`](@ref),
which has a custom backward.
"""
function (m::ModernNCAModel)((x, cand_x, cand_y, y)::Tuple{Any,Any,Any,Any}, ps, st)
    zq, st = _encode(m, x, ps, st)
    p, st, _ = _attend_train(m, zq, vec(y), cand_x, vec(cand_y), ps, st)
    return _finalize(m.loss_type, p), st
end

_train_targets(m::ModernNCAModel, y, idx) =
    _target_block(m, _target_layout(m.loss_type, y), idx)

"""
    _attend_train(m, zq, yq, cand_x, cand_y, ps, st) -> (p, st, (lse, sts))

Online-softmax attention for training. Keys are `zq` itself (diagonal masked)
followed by `cand_x` encoded chunk by chunk with the current `ps` in train mode.
Also returns the per-query log-sum-exp and the layer state captured before each
chunk, which the rrule needs to recompute chunks exactly.
"""
function _attend_train(m::ModernNCAModel, zq, yq, cand_x, cand_y, ps, st)
    acc = _softmax_acc(zq, m.outsize)
    acc = _softmax_fold(acc, _scores(m, zq, zq; mask_self=true)[2],
        _train_targets(m, yq, 1:size(zq, 2)))
    sts = Any[]
    for idx in _chunks(size(cand_x, 2), m.cfg.corpus_chunk_size)
        push!(sts, st)
        zc, st = _encode(m, cand_x[:, idx], ps, st)
        acc = _softmax_fold(acc, _scores(m, zq, zc)[2], _train_targets(m, cand_y, idx))
    end
    p, lse = _softmax_result(acc)
    return p, st, (lse, sts)
end

"""
    _score_grads(m, q, k, d, dS) -> (dq, dk)

Gradients of `s = -d/T`, `d = sqrt(‖q-k‖² + ε)` given `dS` `(num_keys, batch)`,
using ∂d/∂q = (q-k)/d. The `max(0, ·)` clamp in `_pairwise_dist` is treated as
inactive.
"""
function _score_grads(m::ModernNCAModel, q, k, d, dS)
    G = dS ./ (d .* _temperature(m))
    return k * G .- q .* sum(G; dims=1), q * G' .- k .* sum(G; dims=2)'
end

"""
    ChainRulesCore.rrule(::typeof(_attend_train), ...)

FlashAttention-style backward: the forward kept only `p` and `lse`; each key
block is recomputed, `dS = P ∘ (Yᵀ dp − D)` with `D = rowsum(p ∘ dp)`, and
candidate chunks are pulled back through the encoder one at a time.
"""
function ChainRulesCore.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(_attend_train),
    m::ModernNCAModel, zq, yq, cand_x, cand_y, ps, st)
    p, st_out, (lse, sts) = _attend_train(m, zq, yq, cand_x, cand_y, ps, st)
    function attend_train_pullback(Δ)
        dp = unthunk(Δ[1])
        D = sum(p .* dp; dims=1)

        d, s = _scores(m, zq, zq; mask_self=true)
        dS = exp.(s .- lse) .* (_train_targets(m, yq, 1:size(zq, 2))' * dp .- D)
        dzq = .+(_score_grads(m, zq, zq, d, dS)...)

        dps = ZeroTangent()
        for (i, idx) in enumerate(_chunks(size(cand_x, 2), m.cfg.corpus_chunk_size))
            cx = cand_x[:, idx]
            zc, enc_pb = rrule_via_ad(cfg, p_ -> first(_encode(m, cx, p_, sts[i])), ps)
            d, s = _scores(m, zq, zc)
            dS = exp.(s .- lse) .* (_train_targets(m, cand_y, idx)' * dp .- D)
            dq, dk = _score_grads(m, zq, zc, d, dS)
            dzq = dzq .+ dq
            dps = dps + enc_pb(dk)[2]
        end
        return NoTangent(), NoTangent(), dzq, NoTangent(), NoTangent(), NoTangent(), dps, NoTangent()
    end
    return (p, st_out, (lse, sts)), attend_train_pullback
end
