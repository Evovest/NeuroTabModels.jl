# Losses

Losses are concrete callable types, following the [Lux loss API](https://lux.csail.mit.edu/stable/api/Lux/utilities#Loss-Functions): they take `(model, ps, st, data)` and return `(scalar_loss, updated_state, NamedTuple())`.

## Prediction Shape

All losses expect 3D predictions: `(outsize, K, batch)` where `K` is the ensemble size. 2D outputs are reshaped to `(outsize, 1, batch)` automatically.

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
| `:logloss` | `LogLoss` | `(1, K, B)` | `{0, 1}` | raw logits |
| `:mlogloss` | `MLogLoss` | `(C, K, B)` | `{1, …, C}` | raw logits |
| `:gaussian_mle` | `GaussianMLE` | `(2, K, B)` | scalar | `pred[1,:,:]` = μ, `pred[2,:,:]` = log-σ |
| `:tweedie` | `Tweedie` | `(1, K, B)` | non-negative | log-scale pred, ρ = 1.5 |
| `:pearson` | `Pearson` | `(1, K, B)` | scalar | negative Pearson correlation |

## Multiple Targets

With `T` targets (a vector `target_name`), each loss's output rows repeat per target: the prediction is `(T·outsize, K, B)` and the target `(T, B)`. For `:gaussian_mle`, the rows interleave per target as in EvoTrees, μ₁, log-σ₁, μ₂, log-σ₂, …, so `fit` returns `(nobs, 2T)` with the means in odd columns and the standard deviations in even ones. An offset is either one column added to every output or one column per output in that same order.

`:pearson` correlates each target on its own and takes the mean over targets, in the loss and the metric; with two outputs per target the metric reads the μ rows. As in EvoTrees, `:mlogloss` does not support several targets.

## Data Tuples

| Tuple | Contents |
|-------|----------|
| `(x, y)` | standard training |
| `(x, y, w)` | weighted training |
| `(x, y, w, offset)` | with offset (e.g. boosting) |

## Signature

```julia
loss(model, ps, st, data) → (scalar_loss, updated_state, NamedTuple())
```
