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

"""Supertype for column-wise numerical-feature embeddings."""
abstract type AbstractNumericalEmbedding end

"""Supertype for time-column embeddings."""
abstract type AbstractTemporalEmbedding end

"""Top-level embedding spec held by the learner."""
abstract type AbstractEmbedding end

"""No-op embedding: passes raw features through unchanged."""
struct IdentityEmbedding <: AbstractEmbedding end

"""    LinearEmbeddings(; d_embedding=16, activation=:relu)"""
struct LinearEmbeddings <: AbstractNumericalEmbedding
    d_embedding::Int
    activation::Symbol
end
function LinearEmbeddings(; d_embedding::Int=16, activation::Symbol=:relu)
    d_embedding > 0 || throw(ArgumentError("d_embedding must be > 0, got $d_embedding"))
    LinearEmbeddings(d_embedding, activation)
end

"""    PeriodicEmbeddings(; d_embedding=16, frequencies=32, frequencies_init_scale=0.01f0, activation=:relu, lite=false)"""
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

"""    PiecewiseLinearEmbeddings(; d_embedding=16, bins=32, activation=:identity, version=:B)

Requires training data to compute bin edges."""
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

"""    BatchNormEmbeddings()"""
struct BatchNormEmbeddings <: AbstractNumericalEmbedding end

"""    TemporalEmbeddings(; index, order=[4,1,7,0], periods=_DEFAULT_TEMPORAL_PERIODS, trend=true, d_embedding=16)

Fourier features on a single time column (`index` is 1-based into `feature_names`)."""
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

"""    EmbeddingLayer(; num=nothing, temp=nothing)
    EmbeddingLayer(num::AbstractNumericalEmbedding; temp=nothing)
    EmbeddingLayer(d::AbstractDict)

Two typed slots: numerical and temporal. When both are set, the temporal column
(at `temp.index`) is embedded by `temp` and the rest by `num`; outputs are concatenated.

The `AbstractDict` form mirrors the `arch_name`/`arch_config` mechanism: the
numerical type is selected by `:embedding_type` and built from the remaining
keys (extra/`nothing` keys are ignored so a superset hyper-param Dict is safe).
An optional `:temporal` key holds a `Dict` of `TemporalEmbeddings` kwargs."""
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

# keep only keys the target ctor accepts; drop `nothing` so the ctor default applies
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

temporal_out_dim(t::TemporalEmbeddings) = t.d_embedding + (t.trend ? 1 : 0)
_num_d_embedding(s::Union{LinearEmbeddings,PeriodicEmbeddings,PiecewiseLinearEmbeddings}) = s.d_embedding

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
    build_embedding_chain(spec, nfeats; x_train=nothing) -> embedding chain

Builds the input embedding as a numerical core, optionally augmented by a
temporal branch. There are four cases:

  - `IdentityEmbedding` / empty `EmbeddingLayer`: pass features through unchanged.
  - numerical only: the numerical embedding builds the core chain.
  - temporal only (single time column): the time column is embedded on its own.
  - numerical + temporal: the numerical core embeds the feature columns and the
    time column is embedded separately, then concatenated as a parallel branch
    (`TemporalAugmentedEmbeddings`).

The chain emits a flat `(width, batch)` output; its width is recovered with
[`embedding_width`](@ref) rather than returned here.
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

Flattened output width of an embedding chain: the product of the embedding
layer's own analytic `Lux.outputsize` (a `FlattenLayer` collapses all non-batch
dims, so the width is their product). Computed without tracing a forward pass."""
embedding_width(layer, x, rng::AbstractRNG) = prod(Lux.outputsize(layer, x, rng))
embedding_width(::WrappedFunction, x, ::AbstractRNG) = size(x, 1)
embedding_width(c::Chain, x, rng::AbstractRNG) = prod(Lux.outputsize(first(c.layers), x, rng))

"""    has_real_embedding(spec) -> Bool

Whether the spec applies a non-identity embedding."""
has_real_embedding(::IdentityEmbedding) = false
has_real_embedding(e::EmbeddingLayer) = e.num !== nothing || e.temp !== nothing

"""    per_feature_widths(spec, nfeats) -> Vector{Int}

How many output dimensions each input feature expands to, summing to the embedding
output width. A neutral feature-grouping fact; consumers (e.g. TabM's grouped
ensemble scaling init) decide what to do with it."""
per_feature_widths(::IdentityEmbedding, nfeats::Int) = fill(1, nfeats)
function per_feature_widths(e::EmbeddingLayer, nfeats::Int)
    isnothing(e.num) && isnothing(e.temp) && return fill(1, nfeats)
    isnothing(e.temp) && return e.num isa BatchNormEmbeddings ?
        fill(1, nfeats) : fill(_num_d_embedding(e.num), nfeats)
    isnothing(e.num) && return Int[temporal_out_dim(e.temp)]
    d_emb = e.num isa BatchNormEmbeddings ? 1 : _num_d_embedding(e.num)
    return vcat(fill(d_emb, nfeats - 1), Int[temporal_out_dim(e.temp)])
end