# Losses

Losses are concrete callable types, following the [Lux loss API](https://lux.csail.mit.edu/stable/api/Lux/utilities#Loss-Functions): they take `(model, ps, st, data)` and return `(scalar_loss, updated_state, NamedTuple())`.

## Prediction Shape

All losses expect 3D predictions: `(outsize, K, batch)` where `K` is the ensemble size. Other shapes are rejected.

`reduce_pred` averages over `K` on raw predictions before any transformation.

## Usage

```julia
MSE()                    # callable loss functor
LossType(:mse)           # same, from a symbol (`MSE()`)
MSE()(model, ps, st, data)
```

`NeuroTabRegressor` / `NeuroTabClassifier` still take `loss=:mse` (a symbol). `fit` converts that once to a functor and stores it on `NeuroTabModel.loss`.

## Supported Losses

| Symbol | Type | Pred shape | Target | Notes |
|--------|------|-----------|--------|-------|
| `:mse` | `MSE` | `(1, K, B)` | scalar | |
| `:mae` | `MAE` | `(1, K, B)` | scalar | |
| `:huber` | `Huber` | `(1, K, B)` | scalar | δ = 1; quadratic within δ, linear beyond |
| `:logloss` | `LogLoss` | `(1, K, B)` | `{0, 1}` | raw logits |
| `:mlogloss` | `MLogLoss` | `(C, K, B)` | `{1, …, C}` | raw logits |
| `:gaussian_mle` | `GaussianMLE` | `(2, K, B)` | scalar | `pred[1,:,:]` = μ, `pred[2,:,:]` = log-σ |
| `:tweedie` | `Tweedie` | `(1, K, B)` | non-negative | log-scale pred, ρ = 1.5 |
| `:poisson` | `Poisson` | `(1, K, B)` | non-negative | log-scale pred, Poisson deviance |
| `:pearson` | `Pearson` | `(1, K, B)` | scalar | negative Pearson correlation |

Huber's threshold is on the training scale: with the default `scale_target=true`,
δ = 1 corresponds to one target standard deviation.

## Multiple Targets

With `T` targets (a vector `target_name`), the prediction is `(T·outsize, K, B)` and the target `(T, B)`. `:gaussian_mle` rows interleave per target (μ₁, log-σ₁, μ₂, log-σ₂, …), `:pearson` takes the mean correlation over targets, and `:mlogloss` does not support several targets.

## Eval metrics

`metric` defaults to the training loss and must be one of the supported losses. `:pearson` reads the μ rows when the loss is `:gaussian_mle`.

The reported value is one score for the full evaluation set. Each batch contributes a sum, divided by the total weight, so unequal batch sizes do not change it. Pad rows have weight 0 and do not count.

`:pearson` is the exception. Correlation is not a sum of per-row scores, so the reported value is the size-weighted mean of the per-batch or per-group correlations, not one correlation on the concatenated evaluation set. Pads are still dropped inside each batch.

## Data Tuples

The loss sees a 2-, 3-, or 4-tuple. `x` is the model input. A grouped loader wraps it as `(features, mask)` and always includes `w`.

| Tuple | When |
|-------|------|
| `(x, y)` | ungrouped, no `weight_name`, no `offset_name` |
| `(x, y, w)` | `weight_name`, or any grouped batch |
| `(x, y, w, offset)` | `offset_name`. Ungrouped batches with no `weight_name` use `w = 1` |

See [Grouped padding and masks](@ref) for `w` and `mask` on a grouped batch.

## Signature

```julia
loss(model, ps, st, data) → (scalar_loss, updated_state, NamedTuple())
```
