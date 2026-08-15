"""
Default Fourier base periods for `TemporalEmbeddings`, in seconds:
year (`365.25 * 86_400`), 30-day month, week, day. Assumes the time column is
encoded as a POSIX-style seconds-since-epoch float.
"""
const _DEFAULT_TEMPORAL_PERIODS = Float32[31_557_600, 2_629_800, 604_800, 86_400]

"""Fixed Fourier features `sin(ωᵢ t), cos(ωᵢ t)` per harmonic; `ωᵢ = 2πk/T`."""
struct TemporalPeriodic <: LuxCore.AbstractLuxLayer
    omega::Vector{Float32}
end

LuxCore.initialparameters(::AbstractRNG, ::TemporalPeriodic) = (;)

# `omega` is fixed (no gradient), but it lives in `st` rather than as a struct
# field so that `dev(st)` transfers it to the GPU alongside the learnable
# state. Reshaped to a column so the broadcast against `(1, batch)` x is unambiguous.
LuxCore.initialstates(::AbstractRNG, l::TemporalPeriodic) =
    (omega = Matrix{Float32}(reshape(l.omega, length(l.omega), 1)),)

function (l::TemporalPeriodic)(x::AbstractMatrix, ps, st)
    z = st.omega .* x
    return vcat(sin.(z), cos.(z)), st
end

"""Realized temporal embedding: Fourier features + dense projection (+ optional linear trend)."""
struct _TemporalEmbeddings{P,D,trend} <: LuxCore.AbstractLuxContainerLayer{(:periodic, :dense)}
    periodic::P
    dense::D
    t_mean::Float32
    t_std::Float32
end

"""
    _TemporalEmbeddings(t_mean, t_std, order, trend, d_embedding; periods)

Realize a `_TemporalEmbeddings` layer from config.

`order` and `periods` are aligned per-band: for each `(o_i, p_i)`, the Fourier
basis contributes harmonics `k = 1:o_i` with angular frequency `ω = 2πk/p_i`.
All `ω` are concatenated into a single flat vector, so the resulting
`TemporalPeriodic` outputs `2 * sum(order)` features (sin + cos per harmonic),
which the `Dense` then projects to `d_embedding`.

`trend` is lifted into the type parameter so the forward dispatches statically
on whether the standardised raw time `(x - t_mean) / t_std` is appended.
"""
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

LuxCore.outputsize(m::_TemporalEmbeddings{P,D,true},  x, ::AbstractRNG) where {P,D} = (m.dense.out_dims + 1,)
LuxCore.outputsize(m::_TemporalEmbeddings{P,D,false}, x, ::AbstractRNG) where {P,D} = (m.dense.out_dims,)