"""
    BatchNormEmbeddings(nfeats)

Feature-wise `BatchNorm` on `(nfeats, batch)` input. Output shape matches input.
"""
struct BatchNormEmbeddings{L} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    layer::L
end

BatchNormEmbeddings(nfeats::Int) = BatchNormEmbeddings(BatchNorm(nfeats))

function (l::BatchNormEmbeddings)(x::AbstractMatrix, ps, st)
    return l.layer(x, ps, st)
end
