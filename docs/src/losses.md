# Losses

## Prediction Shape

All losses expect 3D predictions: `(outsize, K, batch)` where `K` is the ensemble size. 2D outputs are reshaped to `(outsize, 1, batch)` automatically.

`reduce_pred` averages over `K` on raw predictions before any transformation.

## Usage
```julia
get_loss_fn(:mse)       # by symbol
get_loss_fn(MSE)        # by type
get_loss_type(:mse)     # → MSE
```

## Supported Losses

| Symbol | Type | Pred shape | Target | Notes |
|--------|------|-----------|--------|-------|
| `:mse` | `MSE` | `(1, K, B)` | scalar | |
| `:mae` | `MAE` | `(1, K, B)` | scalar | |
| `:logloss` | `LogLoss` | `(1, K, B)` | `{0, 1}` | raw logits |
| `:mlogloss` | `MLogLoss` | `(C, K, B)` | `{1, …, C}` | raw logits |
| `:gaussian_mle` | `GaussianMLE` | `(2, K, B)` | scalar | `pred[1,:,:]` = μ, `pred[2,:,:]` = log-σ |
| `:tweedie` | `Tweedie` | `(1, K, B)` | non-negative | log-scale pred, ρ = 1.5 |

## Data Tuples

| Tuple | Contents |
|-------|----------|
| `(x, y)` | standard training |
| `(x, y, w)` | weighted training |
| `(x, y, w, offset)` | with offset (e.g. boosting) |

## Signature
```julia
loss_fn(model, ps, st, data) → (scalar_loss, updated_state, NamedTuple())
```