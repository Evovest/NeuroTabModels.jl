"""
    Periodic(nfeats, n_frequencies, sigma)

Sinusoidal encoding `[cos(2π w x), sin(2π w x)]`. Output shape `(2 * n_frequencies, nfeats, batch)`.

# Arguments
- `nfeats::Int`: Number of input features.
- `n_frequencies::Int`: Frequency components per feature.
- `sigma::Float32`: Std-dev for frequency weight init (clamped to ±3σ).
"""
struct Periodic <: LuxCore.AbstractLuxLayer
    nfeats::Int
    n_frequencies::Int
    sigma::Float32
end

function LuxCore.initialparameters(rng::AbstractRNG, l::Periodic)
    bound = l.sigma * 3f0
    w = clamp.(l.sigma .* randn(rng, Float32, l.n_frequencies, l.nfeats), -bound, bound)
    w = reshape(2f0 * Float32(π) .* w, l.n_frequencies, l.nfeats, 1)
    return (weight=w,)
end

LuxCore.initialstates(::AbstractRNG, ::Periodic) = (;)

function (l::Periodic)(x::AbstractMatrix, ps, st)
    x_r = reshape(x, 1, size(x, 1), size(x, 2))
    z = ps.weight .* x_r
    return vcat(cos.(z), sin.(z)), st
end

"""
    PeriodicEmbeddings(nfeats, d_embedding=24; frequencies=48,
                       frequencies_init_scale=0.01f0, activation=relu, lite=false)

`Periodic` followed by per-feature linear projection and activation.

# Arguments
- `nfeats::Int`: Number of input features.
- `d_embedding::Int`: Output dimension per feature (default `24`).
- `frequencies::Int`: Sinusoidal components per feature (default `48`).
- `frequencies_init_scale::Float32`: σ for frequency init (default `0.01f0`).
- `activation`: Post-projection activation (default `relu`).
- `lite::Bool`: Use shared `Dense` instead of per-feature `NLinear` (default `false`).
"""
struct PeriodicEmbeddings{P,L,F} <: LuxCore.AbstractLuxContainerLayer{(:periodic, :linear)}
    periodic::P
    linear::L
    activation::F
    lite::Bool
end

function _PeriodicEmbeddings(
    nfeats::Int,
    d_embedding::Int=24;
    frequencies::Int=48,
    frequencies_init_scale::Float32=0.01f0,
    activation=relu,
    lite::Bool=false,
)
    if lite && activation === identity
        error("lite=true requires a non-identity activation function")
    end
    periodic = Periodic(nfeats, frequencies, frequencies_init_scale)
    linear = if lite
        Dense(2 * frequencies => d_embedding)
    else
        NLinear(nfeats, 2 * frequencies, d_embedding)
    end
    return _PeriodicEmbeddings(periodic, linear, activation, lite)
end

function (m::_PeriodicEmbeddings)(x::AbstractMatrix, ps, st)
    h, st_p = m.periodic(x, ps.periodic, st.periodic)

    h_lin, st_l = if m.lite
        d_in, n, b = size(h)
        h_flat = reshape(h, d_in, n * b)
        out_flat, st_sub = m.linear(h_flat, ps.linear, st.linear)
        reshape(out_flat, size(out_flat, 1), n, b), st_sub
    else
        m.linear(h, ps.linear, st.linear)
    end

    return m.activation.(h_lin), (periodic=st_p, linear=st_l)
end
