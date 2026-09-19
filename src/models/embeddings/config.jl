const act_dict = Dict(
    :identity => identity,
    :relu => relu,
    :tanh => tanh,
    :softplus => softplus,
    :hardtanh => hardtanh,
    :tanhshrink => tanhshrink,
)

# Supertype for embedding configs the learner can hold (`EmbeddingLayer`, numerical embeddings).
abstract type AbstractEmbedding end

# Column-wise numerical-feature embeddings; also the type of an `EmbeddingLayer`'s `num`.
abstract type AbstractNumericalEmbedding <: AbstractEmbedding end

# Time-column embeddings.
abstract type AbstractTemporalEmbedding end

"""
    IdentityEmbedding()

No-op numerical embedding that passes features through unchanged; the default `num` for
an [`EmbeddingLayer`](@ref).
"""
struct IdentityEmbedding <: AbstractNumericalEmbedding end

"""
    LinearEmbeddings(; d_embedding=16, activation=:relu)

Per-feature linear embedding: each feature is projected to `d_embedding` by its own affine map, followed by `activation`.

# Arguments
- `d_embedding::Int`: Output dimension per feature (default `16`).
- `activation`: Activation applied after the projection (default `:relu`).
"""
struct LinearEmbeddings <: AbstractNumericalEmbedding
    d_embedding::Int
    activation::Symbol
end
function LinearEmbeddings(; d_embedding::Int=16, activation=:relu)
    d_embedding > 0 || throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    LinearEmbeddings(d_embedding, Symbol(activation))
end

"""
    PeriodicEmbeddings(; d_embedding=16, frequencies=32, frequencies_init_scale=0.01f0,
                       activation=:relu, lite=false)

Per-feature periodic embedding: each feature is expanded with learned sine/cosine components, then projected to `d_embedding` and passed through `activation`.

# Arguments
- `d_embedding::Int`: Output dimension per feature (default `16`).
- `frequencies::Int`: Number of sinusoidal components per feature (default `32`).
- `frequencies_init_scale::Float32`: Std. dev. initializing the frequencies (default `0.01f0`).
- `activation`: Activation applied after the projection (default `:relu`).
- `lite::Bool`: Share the projection across features to cut parameters (default `false`).
"""
struct PeriodicEmbeddings <: AbstractNumericalEmbedding
    d_embedding::Int
    frequencies::Int
    frequencies_init_scale::Float32
    activation::Symbol
    lite::Bool
end
function PeriodicEmbeddings(;
    d_embedding::Int=16, frequencies::Int=32, frequencies_init_scale::Real=0.01f0, activation=:relu, lite::Bool=false
)
    d_embedding > 0 || throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    frequencies > 0 || throw(ArgumentError("frequencies must be > 0, got $frequencies"))
    PeriodicEmbeddings(d_embedding, frequencies, Float32(frequencies_init_scale), Symbol(activation), lite)
end

"""
    PiecewiseLinearEmbeddings(; d_embedding=16, bins=32, activation=:identity, version=:B)

Per-feature piecewise-linear embedding against bin edges computed from the training data, then projected to `d_embedding`. The bin edges are derived at fit time, so this embedding requires `x_train`.

# Arguments
- `d_embedding::Int`: Output dimension per feature (default `16`).
- `bins::Int`: Number of bins, or per-feature bin counts (default `32`).
- `activation`: Activation applied after the projection (default `:identity`).
- `version`: Encoding variant, `:A` or `:B` (default `:B`).
"""
struct PiecewiseLinearEmbeddings <: AbstractNumericalEmbedding
    d_embedding::Int
    nbins::Int
    activation::Symbol
    version::Symbol
end
function PiecewiseLinearEmbeddings(; d_embedding::Int=16, bins::Int=32, activation=:identity, version=:B)
    version = Symbol(version)
    activation = Symbol(activation)
    d_embedding > 0 || throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    version in (:A, :B) || throw(ArgumentError("version must be :A or :B, got :$version"))
    PiecewiseLinearEmbeddings(d_embedding, bins, activation, version)
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
    TemporalEmbeddings(; index, order=[4, 1, 7, 0], periods=_DEFAULT_TEMPORAL_PERIODS,
                       trend=true, d_embedding=16)

Fourier embedding of a single time column: the column at `index` is expanded into multi-scale sine/cosine features at `periods`, projected to `d_embedding`, and optionally augmented with a linear trend. `order` and `periods` align per band: `order[i]` is the harmonic count for `periods[i]`.

# Arguments
- `index::Int`: Required. 1-based position of the time column in `feature_names`.
- `order::Vector{Int}`: Harmonics per period; nonnegative with at least one positive entry (default `[4, 1, 7, 0]`).
- `periods::Vector{Float32}`: Base periods in column units (default `_DEFAULT_TEMPORAL_PERIODS`).
- `trend::Bool`: Append a linear trend term to the embedding (default `true`).
- `d_embedding::Int`: Projection dimension of the periodic features (default `16`).
"""
struct TemporalEmbeddings <: AbstractTemporalEmbedding
    index::Int
    order::Vector{Int}
    periods::Vector{Float32}
    trend::Bool
    d_embedding::Int
end
function TemporalEmbeddings(;
    index::Int,
    order::AbstractVector{<:Integer}=Int[4, 1, 7, 0],
    periods::AbstractVector{<:Real}=_DEFAULT_TEMPORAL_PERIODS,
    trend::Bool=true,
    d_embedding::Int=16,
)
    index >= 1 || throw(ArgumentError("index must be >= 1, got $index"))
    length(order) == length(periods) ||
        throw(ArgumentError("length(order)=$(length(order)) must equal length(periods)=$(length(periods))"))
    all(>=(0), order) && any(>(0), order) ||
        throw(ArgumentError("order must be non-negative with at least one positive entry"))
    d_embedding > 0 || throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    TemporalEmbeddings(index, Vector{Int}(order), Vector{Float32}(periods), trend, d_embedding)
end

"""
    EmbeddingLayer(; num=IdentityEmbedding(), temp=nothing)
    EmbeddingLayer(num::AbstractNumericalEmbedding; temp=nothing)
    EmbeddingLayer(d::AbstractDict)

Numerical embedding over the non-time columns, optionally combined with a temporal
embedding on the column at `temp.index` (1-based position in the feature list). `num`
always exists (defaults to [`IdentityEmbedding`](@ref)); `nothing` normalizes to it.
When `temp` is set the branches are concatenated features-first, temporal-last.

Prefer the keyword constructor so the two branches are explicit:

```julia
EmbeddingLayer(;
    num=LinearEmbeddings(; d_embedding=16),
    temp=TemporalEmbeddings(; index=2, d_embedding=16),  # column 2 is time
)
```

`index` is the time column, not the embedding width and not `nfeats`.
`build_embedding_chain(config, nfeats; x_train)` still needs `nfeats` = number of columns.

The `Dict` form is for the hyperparameter harness: `:embedding_type` (`:linear`,
`:periodic`, `:piecewise`, `:batchnorm`, `:layernorm`, `:identity`) selects the numerical type, an
optional `:temporal` Dict gives [`TemporalEmbeddings`](@ref) kwargs, and unknown or
`nothing` keys are ignored (missing `:embedding_type` → identity).

```julia
EmbeddingLayer(PeriodicEmbeddings(; d_embedding=24))
EmbeddingLayer(; temp=TemporalEmbeddings(; index=1))            # temporal only
EmbeddingLayer(Dict(:embedding_type => :periodic, :temporal => Dict(:index => 1)))
```
"""
struct EmbeddingLayer{N<:AbstractNumericalEmbedding,T<:Union{Nothing,AbstractTemporalEmbedding}} <: AbstractEmbedding
    num::N
    temp::T
end
_as_num(::Nothing) = IdentityEmbedding()
_as_num(n::AbstractNumericalEmbedding) = n
EmbeddingLayer(; num=IdentityEmbedding(), temp=nothing) = EmbeddingLayer(_as_num(num), temp)
EmbeddingLayer(num::AbstractNumericalEmbedding; temp=nothing) = EmbeddingLayer(num, temp)

const _NUM_EMBEDDING_TYPES = Dict{Symbol,Type}(
    :identity => IdentityEmbedding,
    :linear => LinearEmbeddings,
    :periodic => PeriodicEmbeddings,
    :piecewise => PiecewiseLinearEmbeddings,
    :batchnorm => BatchNormEmbeddings,
    :layernorm => LayerNormEmbeddings,
)

# Used by the `_num_from_dict` methods below. Filters a (possibly superset) config
# Dict down to the keyword arguments one embedding constructor accepts; absent and
# `nothing`-valued keys are skipped so the constructor's own defaults apply.
function _pick(d, keys)
    ps = Pair{Symbol,Any}[]
    for k in keys
        haskey(d, k) && d[k] !== nothing && push!(ps, k => d[k])
    end
    return ps
end

_num_from_dict(::Type{IdentityEmbedding}, d) = IdentityEmbedding()
_num_from_dict(::Type{LinearEmbeddings}, d) = LinearEmbeddings(; _pick(d, (:d_embedding, :activation))...)
function _num_from_dict(::Type{PeriodicEmbeddings}, d)
    PeriodicEmbeddings(; _pick(d, (:d_embedding, :frequencies, :frequencies_init_scale, :activation, :lite))...)
end
function _num_from_dict(::Type{PiecewiseLinearEmbeddings}, d)
    PiecewiseLinearEmbeddings(; _pick(d, (:d_embedding, :bins, :activation, :version))...)
end
_num_from_dict(::Type{BatchNormEmbeddings}, d) = BatchNormEmbeddings()
_num_from_dict(::Type{LayerNormEmbeddings}, d) = LayerNormEmbeddings()

function EmbeddingLayer(d::AbstractDict)
    d = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in d)
    num = if haskey(d, :embedding_type) && d[:embedding_type] !== nothing
        etype = Symbol(d[:embedding_type])
        T = get(_NUM_EMBEDDING_TYPES, etype) do
            throw(ArgumentError("unknown :embedding_type $(repr(etype)); valid: $(collect(keys(_NUM_EMBEDDING_TYPES)))"))
        end
        _num_from_dict(T, d)
    else
        IdentityEmbedding()
    end
    temp = if haskey(d, :temporal)
        td = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in d[:temporal])
        TemporalEmbeddings(; td...)
    else
        nothing
    end
    return EmbeddingLayer(num, temp)
end

# Output width of a temporal embedding: d_embedding, +1 when trend is appended.
temporal_out_dim(t::TemporalEmbeddings) = t.d_embedding + (t.trend ? 1 : 0)

# Per-feature output width of a numerical embedding. Identity, BatchNorm and LayerNorm
# keep each feature at width 1; the expanding embeddings emit `d_embedding` columns per feature.
_num_d_embedding(::IdentityEmbedding) = 1
_num_d_embedding(::BatchNormEmbeddings) = 1
_num_d_embedding(::LayerNormEmbeddings) = 1
_num_d_embedding(config::Union{LinearEmbeddings,PeriodicEmbeddings,PiecewiseLinearEmbeddings}) = config.d_embedding

"""
    needs_x_train(config) -> Bool

Whether building `config` requires the training matrix.

# Arguments
- `config::AbstractEmbedding`: The embedding config to inspect.

# Returns
`true` when `config` needs `x_train` (e.g. piecewise-linear bin edges or temporal normalization statistics), `false` otherwise.
"""
needs_x_train(::Nothing) = false
needs_x_train(::IdentityEmbedding) = false
needs_x_train(::AbstractNumericalEmbedding) = false
needs_x_train(::PiecewiseLinearEmbeddings) = true
needs_x_train(::AbstractTemporalEmbedding) = true
needs_x_train(config::EmbeddingLayer) = needs_x_train(config.num) || needs_x_train(config.temp)

# Numerical embeddings: each builds its own chain emitting a flat (width, batch)
# output. Expanding types append their own FlattenLayer; BatchNorm and LayerNorm are
# already 2D and Identity is a no-op. The builder never inspects the type to decide
# how to flatten; adding a new numerical embedding means adding one method here.
# Row (feature-column) selector used by the temporal `Parallel` branches. Named (not a
# closure) so the resulting `WrappedFunction` is concretely typed via `Base.Fix2`.
_select_rows(x::AbstractMatrix, rows) = x[rows, :]

_build_num(::IdentityEmbedding; nfeats::Int, x_train=nothing) = NoOpLayer()
function _build_num(config::LinearEmbeddings; nfeats::Int, x_train=nothing)
    Chain(
        _LinearEmbeddings(; nfeats, d_embedding=config.d_embedding, activation=act_dict[config.activation]),
        FlattenLayer(),
    )
end
function _build_num(config::PeriodicEmbeddings; nfeats::Int, x_train=nothing)
    Chain(
        _PeriodicEmbeddings(;
            nfeats,
            d_embedding=config.d_embedding,
            frequencies=config.frequencies,
            frequencies_init_scale=config.frequencies_init_scale,
            activation=act_dict[config.activation],
            lite=config.lite,
        ),
        FlattenLayer(),
    )
end
function _build_num(config::PiecewiseLinearEmbeddings; nfeats::Int, x_train=nothing)
    x_train === nothing && error("PiecewiseLinearEmbeddings requires x_train to compute bin edges")
    Chain(
        _PiecewiseLinearEmbeddings(;
            bins=compute_bins(x_train; bins=config.nbins),
            d_embedding=config.d_embedding,
            activation=act_dict[config.activation],
            version=config.version,
        ),
        FlattenLayer(),
    )
end
_build_num(::BatchNormEmbeddings; nfeats::Int, x_train=nothing) = _BatchNormEmbeddings(; nfeats)
_build_num(::LayerNormEmbeddings; nfeats::Int, x_train=nothing) = _LayerNormEmbeddings(; nfeats)

function _build_temp(t::TemporalEmbeddings, x_col)
    x_col === nothing && error("TemporalEmbeddings requires x_train for t_mean/t_std")
    t_mean = Float32(mean(x_col))
    t_std = length(x_col) > 1 ? max(Float32(std(x_col)), 1.0f-6) : 1.0f0
    _TemporalEmbeddings(t_mean, t_std, t.order, t.trend, t.d_embedding; periods=t.periods)
end

"""
    build_embedding_chain(config, nfeats; x_train=nothing)

Build the embedding chain for `config`. A numerical-only config embeds every column; with a
temporal branch the input is routed through a `Lux.Parallel`, applying `num` to the non-time
columns and `temp` to the time column, concatenated features-first then temporal.

# Arguments
- `config::AbstractEmbedding`: The embedding config to build.
- `nfeats::Int`: Number of input features.
- `x_train`: Training matrix `(n_samples, nfeats)`, required when `needs_x_train(config)` (default `nothing`).

# Returns
A `Lux` layer emitting a flat `(width, batch)` output; recover the width with [`embedding_width`](@ref).
"""
function build_embedding_chain(config::AbstractNumericalEmbedding, nfeats::Int; x_train=nothing)
    _build_num(config; nfeats, x_train)
end
function build_embedding_chain(config::EmbeddingLayer, nfeats::Int; x_train=nothing)
    # numerical branch only
    isnothing(config.temp) && return _build_num(config.num; nfeats, x_train)

    # numerical + temporal: route the non-time columns through `num` and the time
    # column through `temp`, concatenating features-first / temporal-last. When `num`
    # is IdentityEmbedding this is the temporal-only case (features pass through).
    idx = config.temp.index
    1 <= idx <= nfeats || throw(ArgumentError("temporal index=$idx out of range for nfeats=$nfeats"))
    keep = setdiff(1:nfeats, idx)
    core = _build_num(config.num; nfeats=nfeats - 1, x_train=x_train === nothing ? nothing : x_train[:, keep])
    temp = _build_temp(config.temp, x_train === nothing ? nothing : @view x_train[:, idx])
    return Parallel(
        vcat,
        Chain(WrappedFunction(Base.Fix2(_select_rows, keep)), core),
        Chain(WrappedFunction(Base.Fix2(_select_rows, idx:idx)), temp),
    )
end

"""
    embedding_width(layer, x, rng) -> Int

Output width of `layer`, measured by a single forward pass on the probe `x`. A forward pass
is robust to the layer's internal structure, unlike analytic `outputsize` which cannot see
through a `vcat` connection.

# Arguments
- `layer`: The embedding layer to measure.
- `x::AbstractMatrix`: Probe input of shape `(nfeats, batch)` (`init` passes `(nfeats, 2)`).
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
