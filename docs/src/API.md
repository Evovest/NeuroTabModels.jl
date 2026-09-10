# Training

Training is a learner plus a DataFrame. The learner holds the architecture
config and the training settings. [`fit`](@ref NeuroTabModels.Fit.fit) builds
the Lux chain, runs the loop, and returns a
[`NeuroTabModel`](@ref NeuroTabModels.NeuroTabModel). Predictions come from
calling that fitted model, or from [`infer`](@ref NeuroTabModels.infer).

Architecture configs and how they become a Lux chain are on
[Models](@ref). Losses are on [Losses](@ref). Numerical and temporal
embeddings are on [Embeddings](@ref).

## Learners

[`NeuroTabRegressor`](@ref) and [`NeuroTabClassifier`](@ref) are the
constructors. They take an architecture config and the knobs that stay
fixed for the run:

```julia
using NeuroTabModels, NeuroTabModels.Models

cfg = MLPConfig(; hidden_size=64, stack_size=2)
learner = NeuroTabRegressor(
    cfg;
    loss=:mse,
    nrounds=100,
    lr=1.0f-2,
    batchsize=2048,
    backend=:zygote,
    device=:cpu,
)
```

These live on the learner, not on `fit`:

| Field | Role |
| --- | --- |
| `arch` | Architecture config (`MLPConfig`, …) |
| `embedding_config` | Numerical / temporal embeddings |
| `loss`, `metric` | Training loss and eval metric (symbols; see [Losses](@ref)) |
| `nrounds`, `early_stopping_rounds` | Epoch budget and patience |
| `lr`, `wd`, `batchsize`, `seed` | Optimiser and data |
| `scale_target` | Standardize the target for losses that support it |
| `backend`, `device`, `gpuID` | AD backend (`:zygote`, `:enzyme`, `:reactant`) and device |

`metric` and `early_stopping_rounds` only take effect when `fit` is given
`deval`. Constructors and the MLJ interface are documented with the types
on [Models](@ref).

## Fit

```julia
m = NeuroTabModels.fit(
    learner,
    dtrain;
    feature_names,
    target_name,
    deval=deval,                 # optional: metrics and early stopping
    weight_name=nothing,
    offset_name=nothing,
    group_name=nothing,          # grouped / padded loader
    eval_group_name=group_name,
    print_every_n=10,
)
```

`feature_names` and `target_name` are required. `dtrain` (and `deval`) must
be `<:AbstractDataFrame`.

`group_name` switches the dataloader to one padded group per step. That
is required for attention models that mix across observations in a
group; see [Grouped padding and masks](@ref). `eval_group_name` can differ
from `group_name` when you want grouped eval metrics while training on the
ungrouped loader.

```@docs
NeuroTabModels.Fit.fit
```

## Inference

A fitted `NeuroTabModel` is callable. That is the usual path:

```julia
p = m(dtrain)                    # natural-scale predictions
p_raw = m(dtrain; proj=false)    # model-scale (logits, log-σ, …)
```

[`infer`](@ref NeuroTabModels.infer) is the same function, also overloaded
on an iterable of feature batches. `proj=true` (default) applies the inverse
link and, when training used `scale_target`, undoes target scaling.
`device` / `backend` default to the values stored on the model.

If the model was trained with `group_name`, inference groups the DataFrame
the same way and drops pad slots after the forward.

```@docs
NeuroTabModels.infer
```
