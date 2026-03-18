using Lux
using Random
using NNlib

"""
    LinearEmbeddings(n_features, d_embedding; activation=relu)

Embeds each continuous feature via a learned affine transformation followed by
an activation: `activation(w_j * x_j + b_j)`.
Produces a `(d_embedding, n_features, batch)` tensor.

# Arguments
- `n_features::Int`: Number of input features.
- `d_embedding::Int`: Embedding dimension per feature.
- `activation`: Activation function applied element-wise (default `relu`).
  E.g. `relu`, `tanh`, `identity`.
"""
struct LinearEmbeddings{F} <: Lux.AbstractLuxLayer
    n_features::Int
    d_embedding::Int
    activation::F
end

function LinearEmbeddings(n_features::Int, d_embedding::Int; activation=NNlib.relu)
    return LinearEmbeddings(n_features, d_embedding, activation)
end

function Lux.initialparameters(rng::AbstractRNG, l::LinearEmbeddings)
    limit = Float32(l.d_embedding)^(-0.5f0)
    weight = reshape((rand(rng, Float32, l.d_embedding, l.n_features) .* 2f0 .* limit) .- limit,
                     l.d_embedding, l.n_features, 1)
    bias = reshape((rand(rng, Float32, l.d_embedding, l.n_features) .* 2f0 .* limit) .- limit,
                   l.d_embedding, l.n_features, 1)
    return (weight=weight, bias=bias)
end

Lux.initialstates(::AbstractRNG, ::LinearEmbeddings) = (;)

function (l::LinearEmbeddings)(x::AbstractMatrix, ps, st)
    x_r = reshape(x, 1, size(x, 1), size(x, 2))
    return l.activation.(muladd.(ps.weight, x_r, ps.bias)), st
end