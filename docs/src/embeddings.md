# Embeddings

Embeddings transform raw input columns before they reach a model backbone. Numerical
embeddings operate column-wise, temporal embeddings reserve one time column, and
categorical embeddings the categorical columns; the branches are concatenated.

They do not consume the grouped padding mask: embeddings map each observation
independently, and `MaskedModel` only reattaches `w` at the `*Attn` core. See
[Grouped padding and masks](@ref).

## Available Embeddings

- `IdentityEmbedding`: leaves numerical features unchanged.
- `LinearEmbeddings`: projects each numerical feature to a learned vector.
- `PeriodicEmbeddings`: expands each feature with learned sinusoidal terms before projection.
- `PiecewiseLinearEmbeddings`: computes feature bins from the training data and embeds the resulting piecewise-linear encoding.
- `BatchNormEmbeddings`: batch-normalizes raw numerical features without expanding them.
- `LayerNormEmbeddings`: layer-normalizes raw numerical features without expanding them.
- `TemporalEmbeddings(; feature)`: embeds the one time column `feature` with Fourier features and an optional trend term.
- `CategoricalEmbeddings(; features)`: learns a vector per level of each categorical column in `features`.

## Configuration

`EmbeddingLayer` combines a numerical embedding with optional temporal and categorical
branches. It is built in one of two ways, and a learner's `embedding_config` takes either.

### From embedding objects

Pass each branch as an embedding object, built with its own constructor (see API below):

- `num`: an `AbstractNumericalEmbedding` from the list above (default `IdentityEmbedding()`).
  It embeds every column of `feature_names` not taken by `temp` or `cat`.
- `temp`: a `TemporalEmbeddings`, or `nothing` (default) for no time column.
- `cat`: a `CategoricalEmbeddings`, or `nothing` (default), in which case `fit` embeds every
  `CategoricalVector` column.

```julia
EmbeddingLayer(;
    num=PiecewiseLinearEmbeddings(; bins=32),
    temp=TemporalEmbeddings(; feature=:t, periods=[50.0]),
    cat=CategoricalEmbeddings(; features=[:store]),
)
```

### From a Dict

For hyperparameter search, or a config read from JSON, pass a Dict with the same three keys,
each optional and each a Dict of constructor kwargs:

- `num`: `type` names the numerical embedding (`:identity`, `:linear`, `:periodic`,
  `:piecewise`, `:batchnorm`, `:layernorm`); the other keys are its kwargs. Keys the type
  does not take, and `nothing` values, are ignored, so one Dict can tune several types.
- `temp`: `TemporalEmbeddings` kwargs, `feature` required.
- `cat`: `CategoricalEmbeddings` kwargs, `features` required.

Keys and values may be strings.

```julia
EmbeddingLayer(Dict(
    :num => Dict(:type => :periodic, :d_embedding => 24),
    :temp => Dict(:feature => :t),
    :cat => Dict(:features => [:store]),
))
```

Piecewise-linear and temporal embeddings need training data when the embedding chain is
built. Use `needs_x_train` to check that requirement for a config.

## Missing values

A `missing` numerical feature reaches the model as `NaN`. `LinearEmbeddings`,
`PeriodicEmbeddings` and `PiecewiseLinearEmbeddings` take `missing_token=true` to embed it as
a learned per-feature `[MISS]` vector, following
[MaskTab](https://arxiv.org/abs/2605.11408); without it, `fit` rejects missing numerical
values. Prediction does not check: a missing numerical value in a model fit without the
token gives a `NaN` prediction. A missing categorical value maps to that column's
unseen-level vector.

```julia
EmbeddingLayer(; num=LinearEmbeddings(; missing_token=true))
```

## API

```@autodocs
Modules = [NeuroTabModels.Models.Embeddings]
Private = false
Order = [:type, :function]
```
