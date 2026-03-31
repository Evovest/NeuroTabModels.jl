using NNlib: relu

"""
    EmbeddingConfig(; kwargs...)

Architecture-agnostic configuration for numerical feature embeddings.
Constructed by the user and passed as `embedding_config` to `NeuroTabRegressor`/`NeuroTabClassifier`.

# Arguments
- `embedding_type::Symbol`: `:periodic`, `:linear`, or `:piecewise` (default `:periodic`).
- `d_embedding::Int`: Embedding dimension per feature (default `24`).
- `activation`: Activation function after projection (default `relu` for periodic/linear,
  `identity` for piecewise).
- `bins::Union{Int, Vector{Int}}`: Number of bins for piecewise embeddings (default `48`).
- `frequencies::Int`: Frequency components for periodic embeddings (default `48`).
- `frequencies_init_scale::Float32`: σ for periodic frequencies init (default `0.01f0`).

# Callable
    config(; nfeats, X_train=nothing)-Lux.Chain

Returns a `Chain(embedding_layer, FlattenLayer())` ready to prepend to any backbone.
"""
struct EmbeddingConfig{F}
    embedding_type::Symbol
    d_embedding::Int
    activation::F
    bins::Union{Int,Vector{Int}}
    frequencies::Int
    frequencies_init_scale::Float32
end

function EmbeddingConfig(;
    embedding_type::Symbol=:periodic,
    d_embedding::Int=16,
    activation=nothing,
    bins::Union{Int,Vector{Int}}=32,
    frequencies::Int=32,
    frequencies_init_scale::Float32=0.01f0,
)
    # Default activation depends on embedding type
    if isnothing(activation)
        activation = embedding_type == :piecewise ? identity : relu
    end

    # Override d_embedding for batchnorm to 1 (for proper derivation of the number of input feature to core model block)
    if embedding_type == :batchnorm
        d_embedding = 1
    end

    return EmbeddingConfig(
        embedding_type, d_embedding, activation,
        bins, frequencies, frequencies_init_scale,
    )
end

function (config::EmbeddingConfig)(; nfeats::Int, x_train=nothing)
    emb = if config.embedding_type == :periodic
        PeriodicEmbeddings(nfeats, config.d_embedding;
            frequencies=config.frequencies,
            frequencies_init_scale=config.frequencies_init_scale,
            activation=config.activation)
    elseif config.embedding_type == :linear
        LinearEmbeddings(nfeats, config.d_embedding; activation=config.activation)
    elseif config.embedding_type == :piecewise
        @assert x_train !== nothing "Piecewise embeddings require `x_train` to compute bin edges."
        bins = compute_bins(x_train; bins=config.bins)
        @assert length(bins) == nfeats "Expected $nfeats bin vectors, got $(length(bins))"
        PiecewiseLinearEmbeddings(bins, config.d_embedding; activation=config.activation)
    elseif config.embedding_type == :batchnorm
        BatchNormEmbeddings(nfeats)
    else
        error("Unsupported embedding type: $(config.embedding_type)")
    end
    return Chain(emb, FlattenLayer())
end
