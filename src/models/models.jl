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

abstract type Architecture end

_broadcast_relu(x) = NNlib.relu.(x)

"""
    train_dataloader(arch, m, default, df; kwargs...)

Per-architecture hook: return the dataloader `fit` should use. Default returns
the tabular `default` unchanged; retrieval-style archs override to substitute
their own iterator (e.g. with a candidate corpus attached).

Overrides may write into `m.info` (e.g. to stash a corpus for inference).
"""
train_dataloader(::Architecture, ::Any, data, ::Any; kw...) = data

"""
    build_chain(arch, embed_chain; nfeats, outsize, d_in, kw...)

Per-architecture hook for assembling the Lux chain that `fit` will train.
The embedding chain always exists (identity is `WrappedFunction(identity)`);
`d_in` is supplied by the caller. Default wires `embed_chain` in front of the
architecture's backbone via `Chain(embed_chain, arch(...))`.
"""
build_chain(arch::Architecture, embed_chain; nfeats, outsize, d_in, kw...) =
    Chain(embed_chain, arch(; nfeats=d_in, outsize))

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

include("MOETree/moetrees.jl")
using .MOETrees

include("TabM/TabM.jl")
using .TabM

include("ModernNCA/modernnca.jl")
using .ModernNCA

# include("MLP/mlp.jl")
# using .MLP

# include("ResNet/resnet.jl")
# using .ResNet

end