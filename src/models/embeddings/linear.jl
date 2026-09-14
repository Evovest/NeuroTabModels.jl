"""
    _LinearEmbeddings(; nfeats, d_embedding, activation=relu)

Embeds each continuous feature via a learned affine transformation followed by
an activation: `activation(w_j * x_j + b_j)`.
Produces a `(d_embedding, nfeats, batch)` tensor.

# Arguments
- `nfeats::Int`: Number of input features.
- `d_embedding::Int`: Embedding dimension per feature.
- `activation`: Activation function applied element-wise (default `relu`).
  E.g. `relu`, `tanh`, `identity`.
"""
struct _LinearEmbeddings{G} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    layer::G
end

# Fan-out uniform, matching the previous `_LinearEmbeddings` init (not fan-in rsqrt).
function _embed_affine_init(rng::AbstractRNG, out::Integer, in::Integer, groups::Integer)
    limit = Float32(out)^(-0.5f0)
    return (rand(rng, Float32, out, in, groups) .* 2.0f0 .* limit) .- limit
end

_as_feature_groups(x::AbstractMatrix) = reshape(x, 1, size(x, 1), size(x, 2))

function _LinearEmbeddings(; nfeats::Int, d_embedding::Int, activation=NNlib.relu)
    return _LinearEmbeddings(
        GroupedDense(
            1 => d_embedding,
            nfeats,
            activation;
            init_weight=_embed_affine_init,
            init_bias=_embed_affine_init,
        ),
    )
end

function (l::_LinearEmbeddings)(x::AbstractMatrix, ps, st)
    return l.layer(_as_feature_groups(x), ps, st)
end

LuxCore.outputsize(l::_LinearEmbeddings, x, ::AbstractRNG) = (l.layer.out_dims, size(x, 1))
