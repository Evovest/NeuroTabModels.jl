using Lux: Chain, FlattenLayer, WrappedFunction
using NNlib: relu, tanh, softplus, hardtanh, tanhshrink

const act_dict = Dict(
    :identity => identity,
    :relu => relu,
    :tanh => tanh,
    :softplus => softplus,
    :hardtanh => hardtanh,
    :tanhshrink => tanhshrink,
)

"""
    AbstractNumericalEmbedding

Supertype for column-wise numerical-feature embeddings.
"""
abstract type AbstractNumericalEmbedding end

"""
    AbstractTemporalEmbedding

Supertype for time-column embeddings.
"""
abstract type AbstractTemporalEmbedding end

"""
    AbstractEmbedding

Supertype for the top-level embedding spec held by the learner.
"""
abstract type AbstractEmbedding end

"""
    IdentityEmbedding()

No-op embedding: passes the raw features through unchanged.
"""
struct IdentityEmbedding <: AbstractEmbedding end

"""
    LinearEmbeddings(; d_embedding=16, activation=:relu)

Per-feature linear embedding. Each numerical feature is projected to a
`d_embedding`-dimensional vector by its own affine map, followed by `activation`.

# Arguments

- `d_embedding=16`: Output dimension per feature.
- `activation=:relu`: Pointwise activation applied after the projection.
"""
struct LinearEmbeddings <: AbstractNumericalEmbedding
    d_embedding::Int
    activation::Symbol
end
function LinearEmbeddings(; d_embedding::Int=16, activation::Symbol=:relu)
    d_embedding > 0 || throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    LinearEmbeddings(d_embedding, activation)
end

"""
    PeriodicEmbeddings(; d_embedding=16, frequencies=32, frequencies_init_scale=0.01f0,
                       activation=:relu, lite=false)

Per-feature periodic embedding. Each feature is expanded with `frequencies` learned
sine/cosine components, then projected to `d_embedding` and passed through `activation`.

# Arguments

- `d_embedding=16`: Output dimension per feature.
- `frequencies=32`: Number of sinusoidal components per feature.
- `frequencies_init_scale=0.01f0`: Std. dev. of the Gaussian initializing the frequencies.
- `activation=:relu`: Pointwise activation applied after the projection.
- `lite=false`: When `true`, share the projection across features to cut parameters.
"""
struct PeriodicEmbeddings <: AbstractNumericalEmbedding
    d_embedding::Int
    frequencies::Int
    frequencies_init_scale::Float32
    activation::Symbol
    lite::Bool
end
function PeriodicEmbeddings(; d_embedding::Int=16, frequencies::Int=32,
                            frequencies_init_scale::Real=0.01f0, activation::Symbol=:relu,
                            lite::Bool=false)
    d_embedding > 0 || throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    frequencies > 0 || throw(ArgumentError("frequencies must be > 0, got $frequencies"))
    PeriodicEmbeddings(d_embedding, frequencies, Float32(frequencies_init_scale),
                       activation, lite)
end

"""
    PiecewiseLinearEmbeddings(; d_embedding=16, bins=32, activation=:identity, version=:B)

Per-feature piecewise-linear embedding. Each feature is encoded against bin edges
computed from the training data, then projected to `d_embedding`. Because the bin
edges are derived at fit time, this embedding requires `x_train`.

# Arguments

- `d_embedding=16`: Output dimension per feature.
- `bins=32`: Number of bins, or a per-feature `Vector{Int}` of bin counts.
- `activation=:identity`: Pointwise activation applied after the projection.
- `version=:B`: Encoding variant, `:A` or `:B`.
"""
struct PiecewiseLinearEmbeddings <: AbstractNumericalEmbedding
    d_embedding::Int
    bins::Union{Int,Vector{Int}}
    activation::Symbol
    version::Symbol
end
function PiecewiseLinearEmbeddings(; d_embedding::Int=16, bins::Union{Int,Vector{Int}}=32,
                                   activation::Symbol=:identity, version::Symbol=:B)
    d_embedding > 0 || throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    version in (:A, :B) ||
        throw(ArgumentError("version must be :A or :B, got :$version"))
    PiecewiseLinearEmbeddings(d_embedding, bins, activation, version)
end

"""
    BatchNormEmbeddings()

Batch-normalize the raw features without expanding their dimension. Each feature
maps to a single output (`d_embedding == 1`).
"""
struct BatchNormEmbeddings <: AbstractNumericalEmbedding end

"""
    TemporalEmbeddings(; index, order=[4, 1, 7, 0], periods=_DEFAULT_TEMPORAL_PERIODS,
                       trend=true, d_embedding=16)

Fourier embedding of a single time column. The column at `index` is expanded into
multi-scale sine/cosine features at the given `periods`, projected to `d_embedding`,
and optionally augmented with a linear trend term.

`order` and `periods` must have equal length: `order[i]` is the number of harmonics
used for `periods[i]`.

# Arguments

- `index`: Required. 1-based position of the time column in `feature_names`.
- `order=[4, 1, 7, 0]`: Harmonics per period; non-negative with at least one positive entry.
- `periods=_DEFAULT_TEMPORAL_PERIODS`: Base periods (in column units) expanded by `order`.
- `trend=true`: When `true`, append a linear trend term to the embedding.
- `d_embedding=16`: Projection dimension of the periodic features.
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
    index >= 1 ||
        throw(ArgumentError("index must be >= 1, got $index"))
    length(order) == length(periods) ||
        throw(ArgumentError("length(order)=$(length(order)) must equal length(periods)=$(length(periods))"))
    all(>=(0), order) && any(>(0), order) ||
        throw(ArgumentError("order must be non-negative with at least one positive entry"))
    d_embedding > 0 ||
        throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    TemporalEmbeddings(index, Vector{Int}(order), Vector{Float32}(periods),
                       trend, d_embedding)
end

"""
    EmbeddingLayer(; num=nothing, temp=nothing)
    EmbeddingLayer(num::AbstractNumericalEmbedding; temp=nothing)
    EmbeddingLayer(d::AbstractDict)

Input embedding combining an optional numerical embedding with an optional temporal
embedding. When both are set, the time column at `temp.index` is embedded by `temp`,
the remaining columns by `num`, and the outputs are concatenated.

The `AbstractDict` form mirrors the `arch_name` / `arch_config` mechanism: the
numerical type is selected by `:embedding_type` and built from the remaining keys.
Unknown or `nothing`-valued keys are ignored, so a superset hyperparameter `Dict` is
accepted. An optional `:temporal` key holds a `Dict` of [`TemporalEmbeddings`](@ref)
keyword arguments.

# Arguments

- `num=nothing`: Numerical embedding, an [`AbstractNumericalEmbedding`](@ref) or `nothing`.
- `temp=nothing`: A [`TemporalEmbeddings`](@ref), or `nothing` for no temporal branch.

# Examples

```julia
# Periodic numerical embedding only
EmbeddingLayer(PeriodicEmbeddings(; d_embedding=24))

# Numerical and temporal, with the time column at position 1
EmbeddingLayer(; num=LinearEmbeddings(), temp=TemporalEmbeddings(; index=1))

# From a (possibly superset) hyperparameter Dict
EmbeddingLayer(Dict(:embedding_type => :periodic, :d_embedding => 24,
                    :temporal => Dict(:index => 1)))
```
"""
struct EmbeddingLayer{
    N<:Union{Nothing,AbstractNumericalEmbedding},
    T<:Union{Nothing,AbstractTemporalEmbedding},
} <: AbstractEmbedding
    num::N
    temp::T
end
EmbeddingLayer(; num=nothing, temp=nothing) = EmbeddingLayer(num, temp)
EmbeddingLayer(num::AbstractNumericalEmbedding; temp=nothing) = EmbeddingLayer(num, temp)

const _NUM_EMBEDDING_TYPES = Dict{Symbol,Type}(
    :linear    => LinearEmbeddings,
    :periodic  => PeriodicEmbeddings,
    :piecewise => PiecewiseLinearEmbeddings,
    :batchnorm => BatchNormEmbeddings,
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

_num_from_dict(::Type{LinearEmbeddings}, d) =
    LinearEmbeddings(; _pick(d, (:d_embedding, :activation))...)
_num_from_dict(::Type{PeriodicEmbeddings}, d) =
    PeriodicEmbeddings(; _pick(d, (:d_embedding, :frequencies, :frequencies_init_scale, :activation, :lite))...)
_num_from_dict(::Type{PiecewiseLinearEmbeddings}, d) =
    PiecewiseLinearEmbeddings(; _pick(d, (:d_embedding, :bins, :activation, :version))...)
_num_from_dict(::Type{BatchNormEmbeddings}, d) = BatchNormEmbeddings()

function EmbeddingLayer(d::AbstractDict)
    d = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in d)
    haskey(d, :embedding_type) ||
        throw(ArgumentError("embedding Dict requires `:embedding_type`"))
    etype = Symbol(d[:embedding_type])
    T = get(_NUM_EMBEDDING_TYPES, etype) do
        throw(ArgumentError("unknown :embedding_type $(repr(etype)); valid: $(collect(keys(_NUM_EMBEDDING_TYPES)))"))
    end
    num = _num_from_dict(T, d)
    temp = if haskey(d, :temporal)
        td = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in d[:temporal])
        TemporalEmbeddings(; td...)
    else
        nothing
    end
    return EmbeddingLayer(num, temp)
end

"""
    temporal_out_dim(t::TemporalEmbeddings) -> Int

Output width of a temporal embedding: `d_embedding`, plus one when `trend` is set.
"""
temporal_out_dim(t::TemporalEmbeddings) = t.d_embedding + (t.trend ? 1 : 0)
_num_d_embedding(s::Union{LinearEmbeddings,PeriodicEmbeddings,PiecewiseLinearEmbeddings}) = s.d_embedding

"""
    needs_x_train(spec) -> Bool

Whether building `spec` requires the training matrix, as for piecewise-linear bin
edges or temporal normalization statistics.
"""
needs_x_train(::Nothing) = false
needs_x_train(::IdentityEmbedding) = false
needs_x_train(::AbstractNumericalEmbedding) = false
needs_x_train(::PiecewiseLinearEmbeddings) = true
needs_x_train(::AbstractTemporalEmbedding) = true
needs_x_train(e::EmbeddingLayer) = needs_x_train(e.num) || needs_x_train(e.temp)

# Numerical embeddings: each builds its own chain emitting a flat (width, batch)
# output. Expanding types append their own FlattenLayer; BatchNorm is already 2D.
# The builder never inspects the type to decide how to flatten; adding a new
# numerical embedding (e.g. categorical) means adding one method here.
_build_num(s::LinearEmbeddings, nfeats::Int, _x) =
    Chain(_LinearEmbeddings(nfeats, s.d_embedding; activation=act_dict[s.activation]), FlattenLayer())
_build_num(s::PeriodicEmbeddings, nfeats::Int, _x) =
    Chain(_PeriodicEmbeddings(nfeats, s.d_embedding;
        frequencies=s.frequencies, frequencies_init_scale=s.frequencies_init_scale,
        activation=act_dict[s.activation], lite=s.lite), FlattenLayer())
function _build_num(s::PiecewiseLinearEmbeddings, nfeats::Int, x_train)
    x_train === nothing &&
        error("PiecewiseLinearEmbeddings requires x_train to compute bin edges")
    Chain(_PiecewiseLinearEmbeddings(compute_bins(x_train; bins=s.bins), s.d_embedding;
        activation=act_dict[s.activation], version=s.version), FlattenLayer())
end
_build_num(::BatchNormEmbeddings, nfeats::Int, _x) = _BatchNormEmbeddings(nfeats)

function _build_temp(t::TemporalEmbeddings, x_col)
    x_col === nothing &&
        error("TemporalEmbeddings requires x_train for t_mean/t_std")
    t_mean = Float32(mean(x_col))
    t_std = length(x_col) > 1 ? max(Float32(std(x_col)), 1f-6) : 1f0
    _TemporalEmbeddings(t_mean, t_std, t.order, t.trend, t.d_embedding; periods=t.periods)
end

"""
    build_embedding_chain(spec, nfeats; x_train=nothing)

Build the embedding chain for `spec`, returning a chain that emits a flat
`(width, batch)` output. The width is recovered with [`embedding_width`](@ref)
rather than returned here.

`IdentityEmbedding` and an empty `EmbeddingLayer` pass the features through
unchanged. A numerical-only spec embeds every column; a temporal-only spec embeds a
single time column on its own; a combined spec embeds the non-time columns numerically
and concatenates the temporal branch as a parallel `TemporalAugmentedEmbeddings`.

# Arguments

- `spec`: An [`AbstractEmbedding`](@ref).
- `nfeats`: Number of input features.
- `x_train=nothing`: Training matrix `(n_samples, nfeats)`, needed if `needs_x_train(spec)`.
"""
build_embedding_chain(::IdentityEmbedding, nfeats::Int; x_train=nothing) = WrappedFunction(identity)
function build_embedding_chain(e::EmbeddingLayer, nfeats::Int; x_train=nothing)
    # no embedding set: pass-through
    isnothing(e.num) && isnothing(e.temp) && return WrappedFunction(identity)

    # numerical only: the core chain
    isnothing(e.temp) && return _build_num(e.num, nfeats, x_train)

    # temporal only: the single time column embedded alone
    if isnothing(e.num)
        nfeats == 1 ||
            throw(ArgumentError("temporal-only embedding requires nfeats == 1 (got $nfeats)"))
        e.temp.index == 1 ||
            throw(ArgumentError("temporal-only: temp.index must be 1 (got $(e.temp.index))"))
        return _build_temp(e.temp, x_train === nothing ? nothing : @view x_train[:, 1])
    end

    # numerical + temporal: core over the other columns, temporal branch concatenated
    idx = e.temp.index
    1 <= idx <= nfeats ||
        throw(ArgumentError("temporal index=$idx out of range for nfeats=$nfeats"))
    keep = setdiff(1:nfeats, idx)
    core = _build_num(e.num, nfeats - 1, x_train === nothing ? nothing : x_train[:, keep])
    temp = _build_temp(e.temp, x_train === nothing ? nothing : @view x_train[:, idx])
    return TemporalAugmentedEmbeddings(core, temp, idx, nfeats)
end

"""
    embedding_width(chain, x, rng) -> Int

Flattened output width of an embedding `chain`, computed analytically with
`Lux.outputsize` without tracing a forward pass. A trailing `FlattenLayer`
collapses all non-batch dimensions, so the width is their product.
"""
embedding_width(layer, x, rng::AbstractRNG) = prod(Lux.outputsize(layer, x, rng))
embedding_width(::WrappedFunction, x, ::AbstractRNG) = size(x, 1)
embedding_width(c::Chain, x, rng::AbstractRNG) = prod(Lux.outputsize(first(c.layers), x, rng))

"""
    has_real_embedding(spec) -> Bool

Whether `spec` applies a non-identity embedding.
"""
has_real_embedding(::IdentityEmbedding) = false
has_real_embedding(e::EmbeddingLayer) = e.num !== nothing || e.temp !== nothing

"""
    per_feature_widths(spec, nfeats) -> Vector{Int}

Output width contributed by each input feature, summing to the total embedding
width. Consumers such as TabM's grouped scaling init use it to keep a feature's
output columns together; the value itself carries no architecture-specific meaning.
"""
per_feature_widths(::IdentityEmbedding, nfeats::Int) = fill(1, nfeats)
function per_feature_widths(e::EmbeddingLayer, nfeats::Int)
    isnothing(e.num) && isnothing(e.temp) && return fill(1, nfeats)
    isnothing(e.temp) && return e.num isa BatchNormEmbeddings ?
        fill(1, nfeats) : fill(_num_d_embedding(e.num), nfeats)
    isnothing(e.num) && return Int[temporal_out_dim(e.temp)]
    d_emb = e.num isa BatchNormEmbeddings ? 1 : _num_d_embedding(e.num)
    return vcat(fill(d_emb, nfeats - 1), Int[temporal_out_dim(e.temp)])
end