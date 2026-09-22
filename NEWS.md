# NeuroTabModels v0.5.0

Breaking release versus v0.4.0. `fit`, functor inference (`m(data)`), and the MLJ interface are unchanged in spirit; several public types, fields, and constructors are not.

## Breaking

**Embeddings.** `EmbeddingConfig` is removed. `NeuroTabRegressor` / `NeuroTabClassifier` now store an `AbstractEmbedding` (`IdentityEmbedding` by default, or `EmbeddingLayer` / a numerical embedding). Pass `nothing`, an `AbstractEmbedding`, or a `Dict` with `:embedding_type`. `EmbeddingConfig(...)` and `isnothing(config.embedding_config)` no longer work.

```julia
# v0.4
embedding_config = EmbeddingConfig(embedding_type=:periodic, d_embedding=24)

# v0.5
embedding_config = Dict(:embedding_type => :periodic, :d_embedding => 24)
# or
embedding_config = EmbeddingLayer(num=PeriodicEmbeddings(; d_embedding=24))
```

Dropped from the embeddings export list: `NLinear`, `Periodic`, `PiecewiseLinearEncoding`, `compute_bins`.

**`NeuroTabModel`.** Field `loss_type::Type{<:LossType}` is now `loss` (a functor instance, e.g. `MSE()`). `m.loss_type` and serialized 0.4.0 models will not load.

**Loss helpers.** `get_loss_fn` / `get_loss_type` are removed. Use `LossType(:mse)` (returns `MSE()`) or the structs directly.

**Architecture call keyword.** `(config)(; nfeats, outsize)` is `(config)(; ins, outsize)` for MLP, NeuroTree, ResNet, TabM, and related backbones. Direct `arch(; nfeats=…)` calls fail.

**Metric.** `metric=:correlation` is rejected; use `:pearson`.

**`TabMConfig`.** `scaling_init` is no longer a config field (unknown kwargs are ignored with a warning).

## Added

- Architectures: `ModernNCAConfig`, `MLPAttnConfig`, `NeuroTreeAttnConfig`.
- Embeddings: `TemporalEmbeddings`, `LayerNormEmbeddings`, `IdentityEmbedding`, `EmbeddingLayer`.
- Loss / metric: Pearson (`:pearson`).
- Mask-aware grouped training and inference (padding masks, `MaskedModel`); grouped predictions are returned in the caller’s row order.
- Attention residual layers and grouped dense building blocks used by the new architectures.
