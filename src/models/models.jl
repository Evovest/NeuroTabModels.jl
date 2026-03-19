module Models

export NeuroTabModel, Architecture
export NeuroTreeConfig, MLPConfig, ResNetConfig, TabMConfig
export Embeddings

using ..Losses
using Lux: Chain
using NNlib

abstract type Architecture end

_broadcast_relu(x) = NNlib.relu.(x)

"""
    NeuroTabModel

A trained neural network model for tabular data. It is the object returned by [`NeuroTabModels.fit`](@ref) and wraps a [Lux.jl](https://lux.csail.mit.edu) `chain` built from one of the supported `Architecture` configurations:

    - NeuroTreeConfig
    - TabMConfig

## Fields

- `loss_type`: the loss function type used during training (e.g. `MSE`, `LogLoss`, `MLogLoss`, `GaussianMLE`)
- `chain`: the underlying `Lux.Chain` neural network
- `info`: a `Dict{Symbol,Any}` storing metadata such as `:feature_names`, `:target_levels`, and `:device`
"""
struct NeuroTabModel{L<:LossType,C}
    loss_type::Type{L}
    chain::C
    info::Dict{Symbol,Any}
end
# @functor NeuroTabModel (chain,)
include("embeddings/embeddings.jl")
using .Embeddings

include("NeuroTree/neurotrees.jl")
using .NeuroTrees

include("TabM/TabM.jl")
using .TabM

# include("MLP/mlp.jl")
# using .MLP

# include("ResNet/resnet.jl")
# using .ResNet

end