module Embeddings

using Lux
using Lux: Chain, FlattenLayer
using LuxCore
using Random
using NNlib
using LuxLib: batched_matmul
import Statistics: quantile

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