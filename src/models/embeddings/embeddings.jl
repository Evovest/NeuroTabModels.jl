module Embeddings

using Lux
using Lux: BatchNorm, Chain, Dense, FlattenLayer
using LuxCore
using NNlib
using Random: AbstractRNG, rand, randn
using Statistics: quantile

export NLinear, LinearEmbeddings
export Periodic, PeriodicEmbeddings
export PiecewiseLinearEncoding, PiecewiseLinearEmbeddings
export compute_bins, EmbeddingConfig

include("compute_bins.jl")
include("nlinear.jl")
include("linear.jl")
include("periodic.jl")
include("piecewise_linear.jl")
include("batchnorm.jl")
include("config.jl")

end
