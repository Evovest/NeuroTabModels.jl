module Models

export NeuroTabModel, get_model_chain

using ..Losses
# import ..NeuroTabModels: get_df_loader_infer
# import ..NeuroTabModels: infer

# import DataFrames: AbstractDataFrame
# import Flux
import Flux: @layer, Chain

abstract type ModelType{T} end

"""
    NeuroTabModel
"""
struct NeuroTabModel{L<:LossType,C<:Chain}
    _loss_type::Type{L}
    chain::C
    info::Dict{Symbol,Any}
end
@layer NeuroTabModel

function get_model_chain(config; nfeats, outsize, kwargs...)
    chain = get_model_chain(ModelType{config.model_type}, config; nfeats, outsize, kwargs...)
    return chain
end

include("NeuroTree/neurotree.jl")
include("MLP/mlp.jl")

end