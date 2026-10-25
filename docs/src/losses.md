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
| `:logloss` | `LogLoss` | `(1, K, B)` | `{0, 1}` | raw logits |
| `:mlogloss` | `MLogLoss` | `(C, K, B)` | `{1, …, C}` | raw logits |
| `:gaussian_mle` | `GaussianMLE` | `(2, K, B)` | scalar | `pred[1,:,:]` = μ, `pred[2,:,:]` = log-σ |
| `:tweedie` | `Tweedie` | `(1, K, B)` | non-negative | log-scale pred, ρ = 1.5 |
| `:poisson` | `Poisson` | `(1, K, B)` | non-negative | log-scale pred, Poisson deviance |
| `:pearson` | `Pearson` | `(1, K, B)` | scalar | negative Pearson correlation |

## Multiple Targets

With `T` targets (a vector `target_name`), the prediction is `(T·outsize, K, B)` and the target `(T, B)`. `:gaussian_mle` rows interleave per target (μ₁, log-σ₁, μ₂, log-σ₂, …), `:pearson` takes the mean correlation over targets, and `:mlogloss` does not support several targets.

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
