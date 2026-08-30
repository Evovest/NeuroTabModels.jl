module Embeddings

using Lux
using LuxCore
using Random
using NNlib
import Statistics: mean, quantile, std

export AbstractNumericalEmbedding, AbstractTemporalEmbedding, AbstractEmbedding
export LinearEmbeddings, PeriodicEmbeddings, PiecewiseLinearEmbeddings
export BatchNormEmbeddings, LayerNormEmbeddings, TemporalEmbeddings, IdentityEmbedding
export EmbeddingLayer, build_embedding_chain, needs_x_train, temporal_out_dim
export embedding_width

using ..GroupedDenseLayer: GroupedDense

include("compute_bins.jl")
include("linear.jl")
include("periodic.jl")
include("piecewise_linear.jl")
include("batchnorm.jl")
include("layernorm.jl")
include("temporal.jl")
include("config.jl")

end