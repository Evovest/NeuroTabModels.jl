# Models

`MLPAttn` / `NeuroTreeAttn` and grouped loaders: see [Grouped padding and masks](@ref)
for why `w` exists, why embeddings never see it, and how `MaskedModel` threads
the mask into the core.

## Layers

`GroupedDense` is the shared per-group linear map used by packed TabM ensembles and by
per-feature numerical embeddings (linear / periodic / piecewise projections).

```@docs
NeuroTabModels.Models.GroupedDense
```

## NeuroTabRegressor

```@docs
NeuroTabRegressor
```

## NeuroTabClassifier

```@docs
NeuroTabClassifier
```

## NeuroTabModel

```@docs
NeuroTabModels.NeuroTabModel
```
