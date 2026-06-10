"""
    _LinearEmbeddings(nfeats, d_embedding; activation=relu)

Embeds each continuous feature via a learned affine transformation followed by
an activation: `activation(w_j * x_j + b_j)`.
Produces a `(d_embedding, nfeats, batch)` tensor.

# Arguments
- `nfeats::Int`: Number of input features.
- `d_embedding::Int`: Embedding dimension per feature.
- `activation`: Activation function applied element-wise (default `relu`).
  E.g. `relu`, `tanh`, `identity`.
"""
struct _LinearEmbeddings{F} <: LuxCore.AbstractLuxLayer
    nfeats::Int
    d_embedding::Int
    activation::F
end

function _LinearEmbeddings(nfeats::Int, d_embedding::Int; activation=NNlib.relu)
    return _LinearEmbeddings(nfeats, d_embedding, activation)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::_LinearEmbeddings)
    limit = Float32(l.d_embedding)^(-0.5f0)
    weight = reshape((rand(rng, Float32, l.d_embedding, l.nfeats) .* 2f0 .* limit) .- limit,
        l.d_embedding, l.nfeats, 1)
    bias = reshape((rand(rng, Float32, l.d_embedding, l.nfeats) .* 2f0 .* limit) .- limit,
        l.d_embedding, l.nfeats, 1)
    return (weight=weight, bias=bias)
end

LuxCore.initialstates(::AbstractRNG, ::_LinearEmbeddings) = (;)

function (l::_LinearEmbeddings)(x::AbstractMatrix, ps, st)
    x_r = reshape(x, 1, size(x, 1), size(x, 2))
    return l.activation.(muladd.(ps.weight, x_r, ps.bias)), st
end

LuxCore.outputsize(l::_LinearEmbeddings, x, ::AbstractRNG) = (l.d_embedding, size(x, 1))