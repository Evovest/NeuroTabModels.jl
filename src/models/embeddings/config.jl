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
- `n_bins::Union{Int, Vector{Int}}`: Number of bins for piecewise embeddings (default `48`).
- `n_frequencies::Int`: Frequency components for periodic embeddings (default `48`).
- `frequency_init_scale::Float32`: σ for periodic frequency init (default `0.01f0`).

# Callable
    config(; nfeats, X_train=nothing)-Lux.Chain

Returns a `Chain(embedding_layer, FlattenLayer())` ready to prepend to any backbone.
"""
struct EmbeddingConfig{F}
    embedding_type::Symbol
    d_embedding::Int
    activation::F
    n_bins::Union{Int,Vector{Int}}
    n_frequencies::Int
    frequency_init_scale::Float32
end

function EmbeddingConfig(;
    embedding_type::Symbol=:periodic,
    d_embedding::Int=24,
    activation=nothing,
    n_bins::Union{Int,Vector{Int}}=48,
    n_frequencies::Int=48,
    frequency_init_scale::Float32=0.01f0,
)
    # Default activation depends on embedding type
    if isnothing(activation)
        activation = embedding_type == :piecewise ? identity : relu
    end

    return EmbeddingConfig(
        embedding_type, d_embedding, activation,
        n_bins, n_frequencies, frequency_init_scale,
    )
end

function (config::EmbeddingConfig)(; nfeats::Int, X_train=nothing)
    emb = if config.embedding_type == :periodic
        PeriodicEmbeddings(nfeats, config.d_embedding;
            n_frequencies=config.n_frequencies,
            frequency_init_scale=config.frequency_init_scale,
            activation=config.activation)
    elseif config.embedding_type == :linear
        LinearEmbeddings(nfeats, config.d_embedding; activation=config.activation)
    elseif config.embedding_type == :piecewise
        @assert X_train !== nothing "Piecewise embeddings require `X_train` to compute bin edges."
        bins = compute_bins(X_train; n_bins=config.n_bins)
        @assert length(bins) == nfeats "Expected $nfeats bin vectors, got $(length(bins))"
        PiecewiseLinearEmbeddings(bins, config.d_embedding; activation=config.activation)
    elseif config.embedding_type == :batchnorm
        # BatchNorm(nfeats)
        BatchNormEmbeddings(nfeats)
    else
        error("Unsupported embedding type: $(config.embedding_type)")
    end
    return Chain(emb, FlattenLayer())
end
