"""
    _LayerNormEmbeddings(; nfeats)

Feature-wise `LayerNorm` on `(nfeats, batch)` input. Output shape matches input.

# Arguments
- `nfeats::Int`: Number of input features.
"""
struct _LayerNormEmbeddings{L} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    layer::L
end

_LayerNormEmbeddings(; nfeats::Int) = _LayerNormEmbeddings(LayerNorm((nfeats,); dims=1))

(l::_LayerNormEmbeddings)(x::AbstractMatrix, ps, st) = l.layer(x, ps, st)

LuxCore.outputsize(l::_LayerNormEmbeddings, x, ::AbstractRNG) = (size(x, 1),)
