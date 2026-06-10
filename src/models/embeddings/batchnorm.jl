using Lux
using Random
using NNlib

"""
    _BatchNormEmbeddings(n_features)

Embeds each continuous feature via a learned affine transformation followed by
an activation: `activation(w_j * x_j + b_j)`.
Produces a `(d_embedding, n_features, batch)` tensor.

# Arguments
- `n_features::Int`: Number of input features.
"""
struct _BatchNormEmbeddings{L} <: Lux.AbstractLuxWrapperLayer{:layer}
    layer::L
end

function _BatchNormEmbeddings(n_features::Int)
    return _BatchNormEmbeddings(BatchNorm(n_features))
end

function (l::_BatchNormEmbeddings)(x::AbstractMatrix, ps, st)
    x_bn, st = l.layer(x, ps, st)
    return x_bn, st
end

Lux.outputsize(l::_BatchNormEmbeddings, x, ::AbstractRNG) = (size(x, 1),)