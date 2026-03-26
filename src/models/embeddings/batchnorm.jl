using Lux
using Random
using NNlib

"""
    BatchNormEmbeddings(n_features)

Embeds each continuous feature via a learned affine transformation followed by
an activation: `activation(w_j * x_j + b_j)`.
Produces a `(d_embedding, n_features, batch)` tensor.

# Arguments
- `n_features::Int`: Number of input features.
"""
struct BatchNormEmbeddings{L} <: Lux.AbstractLuxWrapperLayer{:layer}
    layer::L
end

function BatchNormEmbeddings(n_features::Int)
    return BatchNormEmbeddings(BatchNorm(n_features))
end

function (l::BatchNormEmbeddings)(x::AbstractMatrix, ps, st)
    x_bn, st = l.layer(x, ps, st)
    return x_bn, st
end