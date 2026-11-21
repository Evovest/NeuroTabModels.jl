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
- `TemporalEmbeddings`: embeds one time column with Fourier features and an optional trend term.
- `CategoricalEmbeddings`: learns a vector per level of each categorical column.

## Configuration

`EmbeddingLayer` combines a numerical embedding with optional temporal and categorical
branches. It is built in one of two ways, and a learner's `embedding_config` takes either.

**From embedding objects**, each built with its own constructor:

- `num`: a numerical embedding from the list above; it embeds every column not taken by `temp` or `cat`.
- `temp`: `TemporalEmbeddings(; feature)`, for the one time column `feature`.
- `cat`: `CategoricalEmbeddings(; features)`, for the columns in `features`. Without it, `fit`
  embeds every `CategoricalVector` column.

```julia
EmbeddingLayer(;
    num=PiecewiseLinearEmbeddings(; bins=32),
    temp=TemporalEmbeddings(; feature=:t, periods=[50.0]),
    cat=CategoricalEmbeddings(; features=[:store]),
)
EmbeddingLayer(; num=:piecewise, temp=:t, cat=[:store])   # by name, with default kwargs
```

**From a Dict**, with these keys:

- `:embedding_type`: the numerical embedding, one of `:identity`, `:linear`, `:periodic`,
  `:piecewise`, `:batchnorm`, `:layernorm`. The other top-level keys are its kwargs.
- `:temporal`: a Dict of `TemporalEmbeddings` kwargs.
- `:categorical`: a Dict of `CategoricalEmbeddings` kwargs.

```julia
EmbeddingLayer(Dict(
    :embedding_type => :periodic,
    :d_embedding => 24,
    :temporal => Dict(:feature => :t),
    :categorical => Dict(:features => [:store]),
))
```

Piecewise-linear and temporal embeddings need training data when the embedding chain is
built. Use `needs_x_train` to check that requirement for a config.

## API

```@autodocs
Modules = [NeuroTabModels.Models.Embeddings]
Private = false
Order = [:type, :function]
```
