const act_dict = Dict(
    :identity => identity,
    :relu => relu,
    :tanh => tanh,
    :softplus => softplus,
    :hardtanh => hardtanh,
    :tanhshrink => tanhshrink,
)

function _activation(a)
    a = Symbol(a)
    haskey(act_dict, a) || throw(ArgumentError("unknown activation :$a; valid: $(collect(keys(act_dict)))"))
    return a
end

# Supertype for embedding configs the learner can hold (`EmbeddingLayer`, numerical embeddings).
abstract type AbstractEmbedding end

# Column-wise numerical-feature embeddings; also the type of an `EmbeddingLayer`'s `num`.
abstract type AbstractNumericalEmbedding <: AbstractEmbedding end

# Time-column embeddings.
abstract type AbstractTemporalEmbedding end

# Categorical-feature embeddings.
abstract type AbstractCategoricalEmbedding end

"""
    IdentityEmbedding()

No-op numerical embedding that passes features through unchanged; the default `num` for
an [`EmbeddingLayer`](@ref).
"""
struct IdentityEmbedding <: AbstractNumericalEmbedding end

"""
    LinearEmbeddings(; d_embedding=16, activation=:relu, missing_token=false)

Per-feature linear embedding: each feature is projected to `d_embedding` by its own affine map, followed by `activation`.

# Arguments
- `d_embedding::Int`: Output dimension per feature (default `16`).
- `activation`: Activation applied after the projection (default `:relu`).
- `missing_token::Bool`: Embed a missing value as a learned `[MISS]` vector rather than reject it (default `false`).
"""
struct LinearEmbeddings <: AbstractNumericalEmbedding
    d_embedding::Int
    activation::Symbol
    missing_token::Bool
end
function LinearEmbeddings(; d_embedding::Int=16, activation=:relu, missing_token::Bool=false)
    d_embedding > 0 || throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    LinearEmbeddings(d_embedding, _activation(activation), missing_token)
end

"""
    PeriodicEmbeddings(; d_embedding=16, frequencies=32, frequencies_init_scale=0.01f0,
                       activation=:relu, lite=false, missing_token=false)

Per-feature periodic embedding: each feature is expanded with learned sine/cosine components, then projected to `d_embedding` and passed through `activation`.

# Arguments
- `d_embedding::Int`: Output dimension per feature (default `16`).
- `frequencies::Int`: Number of sinusoidal components per feature (default `32`).
- `frequencies_init_scale::Float32`: Std. dev. initializing the frequencies (default `0.01f0`).
- `activation`: Activation applied after the projection (default `:relu`).
- `lite::Bool`: Share the projection across features to cut parameters (default `false`).
- `missing_token::Bool`: Embed a missing value as a learned `[MISS]` vector rather than reject it (default `false`).
"""
struct PeriodicEmbeddings <: AbstractNumericalEmbedding
    d_embedding::Int
    frequencies::Int
    frequencies_init_scale::Float32
    activation::Symbol
    lite::Bool
    missing_token::Bool
end
function PeriodicEmbeddings(;
    d_embedding::Int=16, frequencies::Int=32, frequencies_init_scale::Real=0.01f0, activation=:relu, lite::Bool=false,
    missing_token::Bool=false,
)
    d_embedding > 0 || throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    frequencies > 0 || throw(ArgumentError("frequencies must be > 0, got $frequencies"))
    PeriodicEmbeddings(
        d_embedding, frequencies, Float32(frequencies_init_scale), _activation(activation), lite, missing_token
    )
end

"""
    PiecewiseLinearEmbeddings(; d_embedding=16, bins=32, activation=:identity, version=:B, missing_token=false)

Per-feature piecewise-linear embedding against bin edges computed from the training data, then projected to `d_embedding`. The bin edges are derived at fit time, so this embedding requires `x_train`.

# Arguments
- `d_embedding::Int`: Output dimension per feature (default `16`).
- `bins::Int`: Number of bins per feature, > 1 (default `32`).
- `activation`: Activation applied after the projection (default `:identity`).
- `version`: Encoding variant, `:A` or `:B` (default `:B`).
- `missing_token::Bool`: Embed a missing value as a learned `[MISS]` vector rather than reject it (default `false`).
"""
struct PiecewiseLinearEmbeddings <: AbstractNumericalEmbedding
    d_embedding::Int
    bins::Int
    activation::Symbol
    version::Symbol
    missing_token::Bool
end
function PiecewiseLinearEmbeddings(;
    d_embedding::Int=16, bins::Int=32, activation=:identity, version=:B, missing_token::Bool=false
)
    version = Symbol(version)
    d_embedding > 0 || throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    bins > 1 || throw(ArgumentError("bins must be > 1, got $bins"))
    version in (:A, :B) || throw(ArgumentError("version must be :A or :B, got :$version"))
    PiecewiseLinearEmbeddings(d_embedding, bins, _activation(activation), version, missing_token)
end

"""
    BatchNormEmbeddings()

Batch-normalize the raw features without expanding their dimension (one output per feature).
"""
struct BatchNormEmbeddings <: AbstractNumericalEmbedding end

"""
    LayerNormEmbeddings()

Layer-normalize the raw features without expanding their dimension (one output per feature).
"""
struct LayerNormEmbeddings <: AbstractNumericalEmbedding end

"""
    TemporalEmbeddings(; feature, order=[4, 1, 7, 0], periods=_DEFAULT_TEMPORAL_PERIODS,
                       trend=true, d_embedding=16)

Fourier embedding of a single time column: the column `feature` is expanded into multi-scale sine/cosine features at `periods`, projected to `d_embedding`, and optionally augmented with a linear trend. `order` and `periods` align per band: `order[i]` is the harmonic count for `periods[i]`.

# Arguments
- `feature`: Required. Name of the time column; must be one of `feature_names`.
- `order::Vector{Int}`: Harmonics per period; nonnegative with at least one positive entry (default `[4, 1, 7, 0]`).
- `periods::Vector{Float32}`: Base periods in column units (default `_DEFAULT_TEMPORAL_PERIODS`).
- `trend::Bool`: Append a linear trend term to the embedding (default `true`).
- `d_embedding::Int`: Projection dimension of the periodic features (default `16`).
"""
struct TemporalEmbeddings <: AbstractTemporalEmbedding
    feature::Symbol
    order::Vector{Int}
    periods::Vector{Float32}
    trend::Bool
    d_embedding::Int
end
function TemporalEmbeddings(;
    feature::Union{Symbol,AbstractString},
    order::AbstractVector=Int[4, 1, 7, 0],
    periods::AbstractVector=_DEFAULT_TEMPORAL_PERIODS,
    trend::Bool=true,
    d_embedding::Int=16,
)
    length(order) == length(periods) ||
        throw(ArgumentError("length(order)=$(length(order)) must equal length(periods)=$(length(periods))"))
    all(>=(0), order) && any(>(0), order) ||
        throw(ArgumentError("order must be non-negative with at least one positive entry"))
    d_embedding > 0 || throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    TemporalEmbeddings(Symbol(feature), Vector{Int}(order), Vector{Float32}(periods), trend, d_embedding)
end

"""
    CategoricalEmbeddings(; features, d_embedding=16)

Learned embedding of categorical columns: each level of each column in `features` maps to
its own `d_embedding` vector. Levels are taken from the training data at fit time; a level
not seen then maps to a per-column "unseen" vector.

# Arguments
- `features`: Required. Names of the categorical columns; each must be one of `feature_names`.
- `d_embedding::Int`: Output dimension per feature (default `16`).
"""
struct CategoricalEmbeddings <: AbstractCategoricalEmbedding
    features::Vector{Symbol}
    d_embedding::Int
end
function CategoricalEmbeddings(; features::AbstractVector, d_embedding::Int=16)
    features = Symbol.(features)
    isempty(features) && throw(ArgumentError("features must not be empty"))
    allunique(features) || throw(ArgumentError("features has duplicate names"))
    d_embedding > 0 || throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    CategoricalEmbeddings(features, d_embedding)
end

_features(::Nothing) = Symbol[]
_features(t::TemporalEmbeddings) = [t.feature]
_features(c::CategoricalEmbeddings) = c.features

"""
    EmbeddingLayer(; num=IdentityEmbedding(), temp=nothing, cat=nothing)
    EmbeddingLayer(config::AbstractDict)

Embedding of each feature by its kind, concatenated numerical first, then temporal, then
categorical. `temp` and `cat` name the columns they embed; `num` embeds every other column
of `feature_names`.

# Arguments
- `num::AbstractNumericalEmbedding`: Numerical embedding (default [`IdentityEmbedding`](@ref)).
- `temp`: A [`TemporalEmbeddings`](@ref), or `nothing` for no temporal column (default).
- `cat`: A [`CategoricalEmbeddings`](@ref), or `nothing` (default), in which case `fit` embeds
  every `CategoricalVector` column.
- `config::AbstractDict`: The same three parts as Dicts of constructor kwargs, each optional:
  - `num`: `type` names the embedding (`:identity`, `:linear`, `:periodic`, `:piecewise`,
    `:batchnorm`, `:layernorm`); the other keys are its kwargs. Keys the type does not take,
    and `nothing` values, are ignored, so one Dict can tune several types.
  - `temp`: [`TemporalEmbeddings`](@ref) kwargs, `feature` required.
  - `cat`: [`CategoricalEmbeddings`](@ref) kwargs, `features` required.
  Keys and values may be strings, so a config read back from JSON builds the same layer.

# Examples
```julia
EmbeddingLayer(;
    num=PeriodicEmbeddings(; d_embedding=24),
    temp=TemporalEmbeddings(; feature=:date),
    cat=CategoricalEmbeddings(; features=[:store]),
)

EmbeddingLayer(Dict(
    :num => Dict(:type => :periodic, :d_embedding => 24),
    :temp => Dict(:feature => :date),
    :cat => Dict(:features => [:store]),
))
```
"""
struct EmbeddingLayer{
    N<:AbstractNumericalEmbedding,
    T<:Union{Nothing,AbstractTemporalEmbedding},
    C<:Union{Nothing,AbstractCategoricalEmbedding},
} <: AbstractEmbedding
    num::N
    temp::T
    cat::C
    function EmbeddingLayer(num::N, temp::T, cat::C) where {N,T,C}
        isdisjoint(_features(temp), _features(cat)) ||
            throw(ArgumentError("temporal feature :$(temp.feature) is also listed as categorical"))
        new{N,T,C}(num, temp, cat)
    end
end

const _NUM_EMBEDDING_TYPES = Dict{Symbol,Type}(
    :identity => IdentityEmbedding,
    :linear => LinearEmbeddings,
    :periodic => PeriodicEmbeddings,
    :piecewise => PiecewiseLinearEmbeddings,
    :batchnorm => BatchNormEmbeddings,
    :layernorm => LayerNormEmbeddings,
)

EmbeddingLayer(; num=IdentityEmbedding(), temp=nothing, cat=nothing) = EmbeddingLayer(num, temp, cat)

_symbol_keys(d) = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in d)

# The hyperparameter harness passes a superset `num` Dict, so each numerical type takes only
# its own fields; absent and `nothing`-valued keys fall back to the constructor defaults.
function _num_embedding(d)
    name = Symbol(d[:type])
    T = get(_NUM_EMBEDDING_TYPES, name) do
        throw(ArgumentError("unknown numerical embedding type :$name; valid: $(collect(keys(_NUM_EMBEDDING_TYPES)))"))
    end
    return T(; (k => d[k] for k in fieldnames(T) if !isnothing(get(d, k, nothing)))...)
end

function EmbeddingLayer(config::AbstractDict)
    config = _symbol_keys(config)
    num = haskey(config, :num) ? _num_embedding(_symbol_keys(config[:num])) : IdentityEmbedding()
    temp = haskey(config, :temp) ? TemporalEmbeddings(; _symbol_keys(config[:temp])...) : nothing
    cat = haskey(config, :cat) ? CategoricalEmbeddings(; _symbol_keys(config[:cat])...) : nothing
    return EmbeddingLayer(num, temp, cat)
end

"""
    needs_x_train(config) -> Bool

Whether building `config` requires the training matrix.

# Arguments
- `config::AbstractEmbedding`: The embedding config to inspect.

# Returns
`true` when `config` needs `x_train` (e.g. piecewise-linear bin edges or temporal normalization statistics), `false` otherwise.
"""
needs_x_train(::Nothing) = false
needs_x_train(::AbstractNumericalEmbedding) = false
needs_x_train(::PiecewiseLinearEmbeddings) = true
needs_x_train(::AbstractTemporalEmbedding) = true
needs_x_train(config::EmbeddingLayer) = needs_x_train(config.num) || needs_x_train(config.temp)

# Numerical embeddings: each builds its own chain emitting a flat (width, batch)
# output. Expanding types append their own FlattenLayer; BatchNorm and LayerNorm are
# already 2D and Identity is a no-op. The builder never inspects the type to decide
# how to flatten; adding a new numerical embedding means adding one method here.
_build_num(::IdentityEmbedding; nfeats::Int, x_train=nothing) = NoOpLayer()
function _build_num(config::LinearEmbeddings; nfeats::Int, x_train=nothing)
    Chain(
        _with_missing(
            config, _LinearEmbeddings(; nfeats, d_embedding=config.d_embedding, activation=act_dict[config.activation]),
            nfeats,
        ),
        FlattenLayer(),
    )
end
function _build_num(config::PeriodicEmbeddings; nfeats::Int, x_train=nothing)
    Chain(
        _with_missing(config, _PeriodicEmbeddings(;
            nfeats,
            d_embedding=config.d_embedding,
            frequencies=config.frequencies,
            frequencies_init_scale=config.frequencies_init_scale,
            activation=act_dict[config.activation],
            lite=config.lite,
        ), nfeats),
        FlattenLayer(),
    )
end
function _build_num(config::PiecewiseLinearEmbeddings; nfeats::Int, x_train=nothing)
    x_train === nothing && error("PiecewiseLinearEmbeddings requires x_train to compute bin edges")
    Chain(
        _with_missing(config, _PiecewiseLinearEmbeddings(;
            bins=compute_bins(permutedims(x_train); bins=config.bins),
            d_embedding=config.d_embedding,
            activation=act_dict[config.activation],
            version=config.version,
        ), nfeats),
        FlattenLayer(),
    )
end
_build_num(::BatchNormEmbeddings; nfeats::Int, x_train=nothing) = _BatchNormEmbeddings(; nfeats)
_build_num(::LayerNormEmbeddings; nfeats::Int, x_train=nothing) = _LayerNormEmbeddings(; nfeats)

_with_missing(config, layer, nfeats) =
    config.missing_token ? _MissingTokens(layer, config.d_embedding, nfeats) : layer

function _build_temp(t::TemporalEmbeddings, x_col)
    x_col === nothing && error("TemporalEmbeddings requires x_train for t_mean/t_std")
    t_mean = Float32(mean(x_col))
    t_std = length(x_col) > 1 ? max(Float32(std(x_col)), 1.0f-6) : 1.0f0
    _TemporalEmbeddings(t_mean, t_std, t.order, t.trend, t.d_embedding; periods=t.periods)
end

_slot(::Nothing, k) = nothing
_slot(x::NamedTuple, k) = x[k]

"""
    embedding_inputs(config, feature_names, cat_levels)

Model input layout for `config`: `feature_names` when every column is numerical, otherwise a
NamedTuple with one slot per embedding kind in use, in the order numerical, temporal,
categorical. `num` and `temp` hold column names, `cat` holds `name => levels` pairs.

# Arguments
- `config::EmbeddingLayer`: The embedding config.
- `feature_names::AbstractVector{Symbol}`: Input feature names.
- `cat_levels::AbstractDict`: Training levels per categorical feature name.
"""
function embedding_inputs(config::EmbeddingLayer, feature_names::AbstractVector{Symbol}, cat_levels::AbstractDict)
    temp, cat = _features(config.temp), _features(config.cat)
    for f in [temp; cat]
        f in feature_names || throw(ArgumentError("embedding feature :$f is not in feature_names"))
    end
    isempty(temp) && isempty(cat) && return feature_names
    slots = (num=setdiff(feature_names, temp, cat), temp=temp, cat=[f => cat_levels[f] for f in cat])
    return (; (k => v for (k, v) in pairs(slots) if !isempty(v))...)
end

"""
    build_embedding_chain(config, features; x_train=nothing)

Build the embedding chain for `config`. With a vector of feature names every column is
embedded numerically. With a NamedTuple layout from [`embedding_inputs`](@ref) each slot of
the input feeds its own branch, and the outputs are concatenated in slot order.

# Arguments
- `config::AbstractEmbedding`: The embedding config to build.
- `features`: Input feature names, or the NamedTuple layout from [`embedding_inputs`](@ref).
- `x_train`: Training input in the model input layout, required when `needs_x_train(config)` (default `nothing`).

# Returns
A `Lux` layer emitting a flat `(width, batch)` output; recover the width with [`embedding_width`](@ref).
"""
function build_embedding_chain(config::AbstractNumericalEmbedding, feature_names; x_train=nothing)
    build_embedding_chain(EmbeddingLayer(; num=config), feature_names; x_train)
end
function build_embedding_chain(config::EmbeddingLayer, feature_names::AbstractVector; x_train=nothing)
    @assert isnothing(config.temp) && isnothing(config.cat) "pass the embedding_inputs layout"
    return _build_num(config.num; nfeats=length(feature_names), x_train)
end
function build_embedding_chain(config::EmbeddingLayer, features::NamedTuple; x_train=nothing)
    branches = []
    haskey(features, :num) &&
        push!(branches, _build_num(config.num; nfeats=length(features.num), x_train=_slot(x_train, :num)))
    haskey(features, :temp) && push!(branches, _build_temp(config.temp, _slot(x_train, :temp)))
    haskey(features, :cat) && push!(branches, Chain(
        _CategoricalEmbeddings([length(levels) for (_, levels) in features.cat], config.cat.d_embedding),
        FlattenLayer(),
    ))
    # `values` turns the slots into a Tuple, which `Parallel` hands out one per branch
    return Chain(WrappedFunction(values), Parallel(vcat, branches...))
end

"""
    embedding_width(layer, x, rng) -> Int

Output width of `layer`, measured by a single forward pass on the probe `x`. A forward pass
is robust to the layer's internal structure, unlike analytic `outputsize` which cannot see
through a `vcat` connection.

# Arguments
- `layer`: The embedding layer to measure.
- `x`: Probe input in the model input layout.
- `rng::AbstractRNG`: RNG used for parameter setup.

# Returns
The flattened output width as an `Int`.
"""
function embedding_width(layer, x, rng::AbstractRNG)
    ps, st = LuxCore.setup(rng, layer)
    st = LuxCore.testmode(st)
    y, _ = layer(x, ps, st)
    return size(y, 1)
end
