module Models

export NeuroTabModel, Architecture
export NeuroTreeConfig, MLPConfig, ResNetConfig

using ..Losses
import Flux: @layer, Chain
import Functors: @functor

abstract type Architecture end

"""
    NeuroTabModel
"""
struct NeuroTabModel{L<:LossType,C<:Chain}
    _loss_type::Type{L}
    chain::C
    info::Dict{Symbol,Any}
end
@functor NeuroTabModel (chain,)

include("NeuroTree/neurotrees.jl")
using .NeuroTrees
include("MLP/mlp.jl")
using .MLP
include("ResNet/resnet.jl")
using .ResNet

end