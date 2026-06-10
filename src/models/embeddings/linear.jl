"""
    _LinearEmbeddings(nfeats, d_embedding; activation=relu)

Per-feature affine map `activation(w * x + b)`. Output shape `(d_embedding, nfeats, batch)`.

# Arguments
- `nfeats::Int`: Number of input features.
- `d_embedding::Int`: Embedding dimension per feature.
- `activation`: Element-wise activation (default `relu`).
"""
struct LinearEmbeddings{F} <: LuxCore.AbstractLuxLayer
    nfeats::Int
    d_embedding::Int
    activation::F
end

function LinearEmbeddings(nfeats::Int, d_embedding::Int; activation=relu)
    return LinearEmbeddings(nfeats, d_embedding, activation)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::LinearEmbeddings)
    limit = Float32(l.d_embedding)^(-0.5f0)
    weight = reshape((rand(rng, Float32, l.d_embedding, l.nfeats) .* 2f0 .* limit) .- limit,
        l.d_embedding, l.nfeats, 1)
    bias = reshape((rand(rng, Float32, l.d_embedding, l.nfeats) .* 2f0 .* limit) .- limit,
        l.d_embedding, l.nfeats, 1)
    return (weight=weight, bias=bias)
end

LuxCore.initialstates(::AbstractRNG, ::LinearEmbeddings) = (;)

function (l::_LinearEmbeddings)(x::AbstractMatrix, ps, st)
    x_r = reshape(x, 1, size(x, 1), size(x, 2))
    return l.activation.(muladd.(ps.weight, x_r, ps.bias)), st
end
