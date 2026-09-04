# Embeddings

Embeddings transform raw input columns before they reach a model backbone. Numerical
embeddings operate column-wise, while temporal embeddings reserve one column for
time features and concatenate that branch with the remaining numerical features.

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

## Configuration

Use `EmbeddingLayer` to combine a numerical embedding with an optional temporal
branch. A dictionary form is also available for model configuration dictionaries.

```julia
EmbeddingLayer(PeriodicEmbeddings(; d_embedding=24))

EmbeddingLayer(;
    num=PiecewiseLinearEmbeddings(; bins=32),
    temp=TemporalEmbeddings(; index=1),
)

EmbeddingLayer(Dict(
    :embedding_type => :periodic,
    :d_embedding => 24,
    :temporal => Dict(:index => 1),
))
```

Piecewise-linear and temporal embeddings need training data when the embedding chain is
built. Use `needs_x_train` to check that requirement for a config.

## API

```@autodocs
Modules = [NeuroTabModels.Models.Embeddings]
```
