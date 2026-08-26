module Models

export NeuroTabModel, Architecture
export Embeddings, EmbeddingLayer
export LinearEmbeddings, PeriodicEmbeddings, PiecewiseLinearEmbeddings
export BatchNormEmbeddings, TemporalEmbeddings, IdentityEmbedding
export AbstractNumericalEmbedding, AbstractTemporalEmbedding, AbstractEmbedding
export NeuroTreeConfig, MLPConfig, ResNetConfig, TabMConfig, MOETreeConfig, ModernNCAConfig

using ..Losses
using Lux: Chain
using NNlib

"""
    Architecture

Abstract supertype for backbone configuration objects.

Subtypes are functors: call with `(config)(; nfeats, outsize)` to build a `Lux.Chain`.
"""
abstract type Architecture end

_broadcast_relu(x) = NNlib.relu.(x)

const activation_dict = Dict{Symbol,Function}(
    :relu => NNlib.relu, :gelu => NNlib.gelu, :sigmoid => NNlib.sigmoid_fast, :tanh => NNlib.tanh_fast
)

function get_activation(act::Symbol)
    haskey(activation_dict, act) ||
        error("Unknown activation: $act. Supported: $(sort(collect(keys(activation_dict))))")
    return activation_dict[act]
end

"""
    train_dataloader(arch, m, default, df; kwargs...)

Per-architecture hook: return the dataloader `fit` should use. Default returns
the tabular `default` unchanged; retrieval-style archs override to substitute
their own iterator (e.g. with a candidate corpus attached).

Overrides may write into `m.info` (e.g. to stash a corpus for inference).
"""
train_dataloader(::Architecture, ::Any, data, ::Any; kwargs...) = data

"""
    build_chain(arch, embed_chain; ins, outsize)

Utility fucntion for assembling the Lux chain that `fit` will train.
"""
function build_chain(arch::Architecture, embed_chain; ins, outsize, kwargs...)
    Chain(embed_chain, arch(; ins, outsize, kwargs...))
end

"""
    infer_dataloader(chain, info, data, dev, ps, st)

Per-architecture hook: return the per-batch iterator `infer` should use.
Default returns `data` unchanged; retrieval-style archs override to wrap each
batch with extra inputs (e.g. a candidate corpus). Dispatched on
`typeof(m.chain)` so arch modules can override on their concrete model type.
"""
infer_dataloader(::Any, ::Any, data, ::Any, ::Any, ::Any) = data

"""
    eval_dataloader(chain, info, data, dev, ps, st)

Per-architecture hook: return the per-batch iterator the eval `CallBack` should
use. Default returns `data` unchanged; retrieval-style archs override to wrap
each batch's `x` with extra inputs (e.g. a candidate corpus) so that the eval
forward pass matches the training/inference signature. Dispatched on
`typeof(m.chain)`.
"""
eval_dataloader(::Any, ::Any, data, ::Any, ::Any, ::Any) = data

"""
    NeuroTabModel

The object containing the model and associated metadata.

# Fields

- `loss_type`: the loss type used in training (`MSE`, `LogLoss`, `MLogLoss`, `GaussianMLE`).
- `chain`: the underlying `Lux.Chain` neural network.
- `info`: a `Dict{Symbol,Any}` of metadata such as `:feature_names`, `:target_levels`,
  and `:device`, plus the fitted parameters (`ps`) and state (`st`).
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

include("MOETree/moetrees.jl")
using .MOETrees

include("TabM/TabM.jl")
using .TabM

include("MLP/mlp.jl")
using .MLP

include("ResNet/resnet.jl")
using .ResNet

include("ModernNCA/modernnca.jl")
using .ModernNCA

end