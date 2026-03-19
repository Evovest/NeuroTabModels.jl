module Embeddings

using Lux
using LuxCore
using Random
using NNlib
using LuxLib: batched_matmul
import Statistics: quantile

export NLinear, LinearEmbeddings, LinearReLUEmbeddings
export Periodic, PeriodicEmbeddings
export PiecewiseLinearEncoding, PiecewiseLinearEmbeddings
export compute_bins

include("compute_bins.jl")
include("nlinear.jl")
include("linear.jl")
include("periodic.jl")
include("piecewise_linear.jl")

end