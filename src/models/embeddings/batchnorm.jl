"""
    _BatchNormEmbeddings(; nfeats)

Feature-wise `BatchNorm` on `(nfeats, batch)` input. Output shape matches input.

# Arguments
- `nfeats::Int`: Number of input features.
"""
struct _BatchNormEmbeddings{L} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    layer::L
end

_BatchNormEmbeddings(; nfeats::Int) = _BatchNormEmbeddings(BatchNorm(nfeats))

(l::_BatchNormEmbeddings)(x::AbstractMatrix, ps, st) = l.layer(x, ps, st)

LuxCore.outputsize(l::_BatchNormEmbeddings, x, ::AbstractRNG) = (size(x, 1),)