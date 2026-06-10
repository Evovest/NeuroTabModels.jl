const act_dict = Dict(
    :identity => identity,
    :relu => relu,
    :tanh => tanh,
    :softplus => softplus,
    :hardtanh => hardtanh,
    :tanhshrink => tanhshrink,
)

"""
    EmbeddingConfig(; kwargs...)

Configuration for numerical feature embeddings.
Pass as `embedding_config` to `NeuroTabRegressor` / `NeuroTabClassifier`.

# Arguments
- `embedding_type::Symbol`: `:periodic`, `:linear`, `:piecewise`, or `:batchnorm` (default `:periodic`).
- `d_embedding::Int`: Embedding dimension per feature (default `16`; forced to `1` for `:batchnorm`).
- `activation::Symbol`: `:identity`, `:relu`, `:tanh`, etc. (default `:relu`, or `:identity` for `:piecewise`).
- `bins::Union{Int, Vector{Int}}`: Bin count for `:piecewise` (default `32`).
- `frequencies::Int`: Sinusoidal components for `:periodic` (default `32`).
- `frequencies_init_scale::Float32`: σ for periodic frequency init (default `0.01f0`).
"""
struct EmbeddingConfig
    embedding_type::Symbol
    d_embedding::Int
    activation::Symbol
    bins::Union{Int,Vector{Int}}
    frequencies::Int
    frequencies_init_scale::Float32
end

function EmbeddingConfig(;
    embedding_type=:periodic,
    d_embedding::Int=16,
    activation=nothing,
    bins::Union{Int,Vector{Int}}=32,
    frequencies::Int=32,
    frequencies_init_scale::Float32=0.01f0,
)

    embedding_type = Symbol(embedding_type)
    if isnothing(activation)
        activation = embedding_type == :piecewise ? :identity : :relu
    end
    activation = Symbol(activation)

    if embedding_type == :batchnorm
        d_embedding = 1
    end

    return EmbeddingConfig(embedding_type, d_embedding, activation, bins, frequencies, frequencies_init_scale)
end

"""
    (config::EmbeddingConfig)(; nfeats, x_train=nothing)

Build `Chain(embedding, FlattenLayer())` from `config`.

# Arguments
- `nfeats::Int`: Number of input features.
- `x_train`: Training matrix `(n_samples, n_features)`; required for `:piecewise`.
"""
function (config::EmbeddingConfig)(; nfeats::Int, x_train=nothing)
    emb = if config.embedding_type == :periodic
        PeriodicEmbeddings(nfeats, config.d_embedding;
            frequencies=config.frequencies,
            frequencies_init_scale=config.frequencies_init_scale,
            activation=act_dict[config.activation])
    elseif config.embedding_type == :linear
        LinearEmbeddings(nfeats, config.d_embedding; activation=act_dict[config.activation])
    elseif config.embedding_type == :piecewise
        @assert x_train !== nothing "Piecewise embeddings require `x_train` to compute bin edges."
        bins = compute_bins(x_train; bins=config.bins)
        @assert length(bins) == nfeats "Expected $nfeats bin vectors, got $(length(bins))"
        PiecewiseLinearEmbeddings(bins, config.d_embedding; activation=act_dict[config.activation])
    elseif config.embedding_type == :batchnorm
        BatchNormEmbeddings(nfeats)
    else
        error("Unsupported embedding type: $(config.embedding_type)")
    end
    return Chain(emb, FlattenLayer())
end
