const _DEFAULT_TEMPORAL_PERIODS = Float32[31_557_600, 2_629_800, 604_800, 86_400]

"""Fixed Fourier features `sin(ωᵢ t), cos(ωᵢ t)` per harmonic; `ωᵢ = 2πk/T`."""
struct TemporalPeriodic <: Lux.AbstractLuxLayer
    omega::Vector{Float32}
end

Lux.initialparameters(::AbstractRNG, ::TemporalPeriodic) = (;)
Lux.initialstates(::AbstractRNG, l::TemporalPeriodic) =
    (omega = Matrix{Float32}(reshape(l.omega, length(l.omega), 1)),)

function (l::TemporalPeriodic)(x::AbstractMatrix, ps, st)
    z = st.omega .* x
    return vcat(sin.(z), cos.(z)), st
end

"""Realized temporal embedding: Fourier features + dense projection (+ optional linear trend)."""
struct _TemporalEmbeddings{P,D,trend} <: Lux.AbstractLuxContainerLayer{(:periodic, :dense)}
    periodic::P
    dense::D
    t_mean::Float32
    t_std::Float32
end

function _TemporalEmbeddings(
    t_mean::Real, t_std::Real,
    order::AbstractVector{<:Integer}, trend::Bool, d_embedding::Int;
    periods::AbstractVector{<:Real}=_DEFAULT_TEMPORAL_PERIODS,
)
    @assert length(order) == length(periods) "length(order) must equal length(periods)"
    @assert any(>(0), order) "`order` must contain at least one positive entry"
    omega = Float32[2f0 * Float32(π) * Float32(k) / Float32(p)
                    for (o, p) in zip(order, periods) for k in 1:o]
    periodic = TemporalPeriodic(omega)
    dense = Dense(2 * length(omega) => d_embedding, NNlib.relu)
    return _TemporalEmbeddings{typeof(periodic),typeof(dense),trend}(
        periodic, dense, Float32(t_mean), Float32(t_std),
    )
end

function (m::_TemporalEmbeddings{P,D,true})(x::AbstractMatrix, ps, st) where {P,D}
    h, st_p = m.periodic(x, ps.periodic, st.periodic)
    out, st_d = m.dense(h, ps.dense, st.dense)
    return vcat(out, (x .- m.t_mean) ./ m.t_std), (periodic=st_p, dense=st_d)
end

function (m::_TemporalEmbeddings{P,D,false})(x::AbstractMatrix, ps, st) where {P,D}
    h, st_p = m.periodic(x, ps.periodic, st.periodic)
    out, st_d = m.dense(h, ps.dense, st.dense)
    return out, (periodic=st_p, dense=st_d)
end

"""Composes a numerical embedding over `nfeats-1` columns with a temporal embedding
on column `temporal_index`. Output is flattened-numerical concatenated with temporal."""
struct TemporalAugmentedEmbeddings{F,T} <: Lux.AbstractLuxContainerLayer{(:features, :temporal)}
    features::F
    temporal::T
    temporal_index::Int
    nfeats::Int
end

function (l::TemporalAugmentedEmbeddings)(x::AbstractMatrix, ps, st)
    idx, n = l.temporal_index, l.nfeats
    x_time = x[idx:idx, :]
    x_feats = if idx == 1
        x[2:n, :]
    elseif idx == n
        x[1:n-1, :]
    else
        vcat(x[1:idx-1, :], x[idx+1:n, :])
    end
    h_feats, st_f = l.features(x_feats, ps.features, st.features)
    h_time, st_t = l.temporal(x_time, ps.temporal, st.temporal)
    d, k, b = size(h_feats)
    return vcat(reshape(h_feats, d * k, b), h_time), (features=st_f, temporal=st_t)
end
