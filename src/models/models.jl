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

The object containing the model and associated metadata.

## Fields

- `loss_type`: the loss function type used during training (e.g. `MSE`, `LogLoss`, `MLogLoss`, `GaussianMLE`)
- `chain`: the underlying `Lux.Chain` neural network
- `info`: a `Dict{Symbol,Any}` storing metadata such as `:feature_names`, `:target_levels`, `:device`, `logger`, as well as fitted parameters (`ps`) and state (`st`).
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