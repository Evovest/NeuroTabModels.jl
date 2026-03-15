using Lux
using Random
using NNlib

import ...Models: _broadcast_relu

"""
    PiecewiseLinearEncoding(bins)

Non-trainable piecewise-linear encoding using precomputed bin edges.
Output shape `(max_n_bins, n_features, batch)`.

# Arguments
- `bins::Vector{<:AbstractVector}`: Bin edges per feature from [`compute_bins`](@ref).
"""
struct PiecewiseLinearEncoding <: Lux.AbstractLuxLayer
    bins::Vector{Vector{Float32}}
    n_features::Int
    max_n_bins::Int
end

function PiecewiseLinearEncoding(bins::Vector{<:AbstractVector})
    @assert length(bins) > 0
    n_features = length(bins)
    n_bins_list = [length(b) - 1 for b in bins]
    max_n_bins = maximum(n_bins_list)
    bins_f32 = [Float32.(b) for b in bins]
    return PiecewiseLinearEncoding(bins_f32, n_features, max_n_bins)
end

Lux.initialparameters(::AbstractRNG, ::PiecewiseLinearEncoding) = (;)

function Lux.initialstates(::AbstractRNG, l::PiecewiseLinearEncoding)
    M, N = l.max_n_bins, l.n_features

    weight = zeros(Float32, M, N)
    bias   = zeros(Float32, M, N)
    lo     = zeros(Float32, M, N)
    hi     = ones(Float32, M, N)

    for (i, bin_edges) in enumerate(l.bins)
        bin_width = diff(bin_edges)
        w = 1f0 ./ bin_width
        b = -bin_edges[1:end-1] ./ bin_width
        nb = length(bin_edges) - 1

        weight[end, i] = w[end]
        bias[end, i]   = b[end]
        if nb > 1
            weight[1:nb-1, i] = w[1:end-1]
            bias[1:nb-1, i]   = b[1:end-1]
        end

        # Pre-compute per-element clamp bounds matching original activation:
        #   first row  -> min(h, 1)    => lo=-Inf, hi=1
        #   middle     -> clamp(h,0,1) => lo=0,    hi=1  (default)
        #   last row   -> max(h, 0)    => lo=0,    hi=+Inf
        #   single bin -> unclamped    => lo=-Inf, hi=+Inf
        if nb == 1
            lo[end, i] = -Inf32
            hi[end, i] =  Inf32
        else
            lo[1, i]   = -Inf32
            hi[end, i] =  Inf32
        end
    end

    # Pre-reshape to 3D to avoid per-call reshape in forward pass
    return (
        weight = reshape(weight, M, N, 1),
        bias   = reshape(bias,   M, N, 1),
        lo     = reshape(lo,     M, N, 1),
        hi     = reshape(hi,     M, N, 1),
    )
end

function (l::PiecewiseLinearEncoding)(x::AbstractMatrix, ps, st)
    x_r = reshape(x, 1, size(x, 1), size(x, 2))
    h = clamp.(muladd.(st.weight, x_r, st.bias), st.lo, st.hi)
    return h, st
end

"""
    PiecewiseLinearEmbeddings(bins, d_embedding; activation=false, version=:B)

Learnable embeddings on top of `PiecewiseLinearEncoding`.
Version `:A`: PLE -> NLinear (with bias).
Version `:B`: PLE -> NLinear (zero-init, no bias) + LinearEmbeddings residual.
Output shape `(d_embedding, n_features, batch)`.

# Arguments
- `bins::Vector{<:AbstractVector}`: Bin edges per feature from [`compute_bins`](@ref).
- `d_embedding::Int`: Embedding dimension per feature.
- `activation::Bool`: Apply ReLU after projection (default `false`).
- `version::Symbol`: `:A` or `:B` (default `:B`).
"""
struct PiecewiseLinearEmbeddings{L0,I,L} <: Lux.AbstractLuxContainerLayer{(:linear0, :encoding, :linear)}
    linear0::L0
    encoding::I
    linear::L
    use_activation::Bool
    version::Symbol
end

function PiecewiseLinearEmbeddings(
    bins::Vector{<:AbstractVector},
    d_embedding::Int;
    activation::Bool=false,
    version::Symbol=:B,
)
    @assert version in (:A, :B)
    n_features = length(bins)
    max_n_bins = maximum(length(b) - 1 for b in bins)

    encoding = PiecewiseLinearEncoding(bins)
    linear0 = (version == :B) ? LinearEmbeddings(n_features, d_embedding) : nothing
    linear = NLinear(n_features, max_n_bins, d_embedding; bias=(version == :A))

    return PiecewiseLinearEmbeddings(linear0, encoding, linear, activation, version)
end

function Lux.initialparameters(rng::AbstractRNG, m::PiecewiseLinearEmbeddings)
    ps_l0 = m.linear0 === nothing ? nothing : Lux.initialparameters(rng, m.linear0)
    ps_enc = Lux.initialparameters(rng, m.encoding)

    if m.version == :B
        n = m.linear.n
        ps_lin = (weight=zeros(Float32, m.linear.out_features, m.linear.in_features, n),)
    else
        ps_lin = Lux.initialparameters(rng, m.linear)
    end

    return (linear0=ps_l0, encoding=ps_enc, linear=ps_lin)
end

function (m::PiecewiseLinearEmbeddings)(x::AbstractMatrix, ps, st)
    val_linear0 = nothing
    st_l0 = st.linear0

    if m.linear0 !== nothing
        val_linear0, st_l0 = m.linear0(x, ps.linear0, st.linear0)
    end

    h_enc, st_enc = m.encoding(x, ps.encoding, st.encoding)
    h_proj, st_lin = m.linear(h_enc, ps.linear, st.linear)

    h_final = val_linear0 === nothing ? h_proj : (val_linear0 .+ h_proj)

    if m.use_activation
        h_final = _broadcast_relu(h_final)
    end

    return h_final, (linear0=st_l0, encoding=st_enc, linear=st_lin)
end