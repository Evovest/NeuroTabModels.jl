# Models

An **architecture** is a Lux compute graph: an `AbstractLuxLayer` (usually a
`Lux.Chain`, sometimes a custom container) that maps feature tensors to
predictions.

Each family is described by a **config**. Calling the config instantiates the
Lux architecture. 

The following names refer to distinct objects:

- **Architecture config** — `MLPConfig`, `NeuroTreeConfig`, and the other
  concrete [`Architecture`](@ref NeuroTabModels.Models.Architecture) types.
  Hyperparameters only.
- **Lux architecture** — the layer returned by `config(; ins, outsize)`. This
  is the compute graph.
- **Fitted model** — a [`NeuroTabModel`](@ref NeuroTabModels.NeuroTabModel)
  produced by training. It stores the trained Lux architecture plus loss and
  metadata.

[`NeuroTabRegressor`](@ref) and [`NeuroTabClassifier`](@ref) are **learners**.
They hold an architecture config together with training settings (loss,
optimizer, device). Training is documented on the [Training](@ref) page.

## Instantiating from a config

A config is constructed, then called with the input width (`ins`) and output
width (`outsize`):

```julia
using NeuroTabModels.Models

cfg = MLPConfig(; hidden_size=64, stack_size=2)
lux_arch = cfg(; ins=10, outsize=1)   # a Lux.Chain
```

`ins` is the width of the tensor that enters the backbone (raw features, or
the embedding width when embeddings are prepended). `outsize` is the
prediction dimension (for example `1` for a scalar regression).

Learners take the **config**, not a pre-built Lux layer:

```julia
learner = NeuroTabRegressor(cfg; loss=:mse, nrounds=10)
```

Configs can also be specified from a dictionary (`arch_name` / `arch_config`)
on the learner constructor; see [`NeuroTabRegressor`](@ref).

Available configs, with their hyperparameters, are on the pages for
[MLP](@ref), [ResNet](@ref), [NeuroTrees](@ref), [TabM](@ref), and
[ModernNCA](@ref). Those families include attention and mixture-of-experts
variants (`MLPAttnConfig`, `NeuroTreeAttnConfig`, `MOETreeConfig`).

## Embeddings and composition

A config builds the **backbone** only. Numerical and temporal embeddings are a
separate Lux chain; see [Embeddings](@ref). Training composes embeddings in
front of the backbone (`Chain(embed, backbone)` in the usual case).

Shared building blocks used inside backbones and embeddings are on
[Layers](@ref). Grouped loaders and attention models that need a padding mask
are on [Grouped padding and masks](@ref); that page also covers
[`MaskedModel`](@ref NeuroTabModels.Models.MaskedModel), which is the
composition used instead of `Chain` when the backbone must see that mask.

## Architecture API

```@docs
NeuroTabModels.Models.Architecture
NeuroTabModels.Models.uses_batch_mask
NeuroTabModels.Models.MaskedModel
```

## Learners and fitted models

```@docs
NeuroTabRegressor
NeuroTabClassifier
NeuroTabModels.NeuroTabModel
```
