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
"""
struct NeuroTabModel{L<:LossType,C}
    _loss_type::Type{L}
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