module Infer

using ..Data
using ..Losses
using ..Models

using Lux
using Lux: cpu_device, gpu_device
using NNlib: sigmoid, softmax
using Statistics: mean
using DataFrames: AbstractDataFrame, GroupedDataFrame, groupby
import MLUtils: DataLoader

export infer, reduce_pred

reduce_pred(pred::AbstractMatrix) = pred
reduce_pred(pred::AbstractArray{T,3}) where {T} = dropdims(mean(pred; dims=2); dims=2)

"""
    _get_device(device::Symbol; gpuID::Integer=0)

Resolve a Lux device from a symbol: `:cpu` or `:gpu`.
`backend=:reactant` requires `using Reactant` to activate `NeuroTabModelsReactantExt`.
"""
_get_device(device::Symbol; gpuID::Integer=0) = _get_device(Val(device); gpuID)
_get_device(backend::Symbol, device::Symbol; gpuID::Integer=0) = _get_device(Val(backend), Val(device); gpuID)
_get_device(::Val, ::Val{D}; gpuID::Integer=0) where {D} = _get_device(Val(D); gpuID)
_get_device(::Val{:cpu}; gpuID::Integer=0) = cpu_device()
_get_device(::Val{:gpu}; gpuID::Integer=0) = gpu_device(gpuID == 0 ? nothing : gpuID)
function _get_device(::Val{D}; gpuID::Integer=0) where {D}
    error("Unsupported `device=:$D`. Supported devices are [:cpu, :gpu].")
end

function _forward_reduce(chain, x, ps, st)
    pred, _ = chain(x, ps, st)
    return reduce_pred(pred)
end

# Assemble raw predictions into final structure (no transforms)
_assemble(::Type{<:MLogLoss}, raw_preds) = reduce(hcat, raw_preds)
_assemble(::Type{<:GaussianMLE}, raw_preds) = reduce(hcat, raw_preds)
_assemble(::Type, raw_preds) = vcat([vec(p) for p in raw_preds]...)

# Apply inverse link to convert from model scale to natural scale
_inverse_link(::Type{<:LogLoss}, pred) = sigmoid.(pred)
_inverse_link(::Type{<:Tweedie}, pred) = exp.(pred)
_inverse_link(::Type{<:Union{MSE,MAE,Correlation}}, pred) = pred
_inverse_link(::Type{<:MLogLoss}, pred) = Matrix(softmax(pred; dims=1)')
function _inverse_link(::Type{<:GaussianMLE}, pred)
    p = Matrix(pred')
    @views p[:, 1] .= p[:, 1]
    @views p[:, 2] .= exp.(p[:, 2])
    return p
end

_scaler(::Type{<:LossType}, p, scalers) = p
_scaler(::Type{<:Union{MSE,MAE,Correlation}}, p, scalers::NamedTuple) = p .* scalers[:sigma] .+ scalers[:mu]
function _scaler(::Type{<:GaussianMLE}, p, scalers::NamedTuple)
    @views p[:, 1] .= p[:, 1] .* scalers[:sigma] .+ scalers[:mu]
    @views p[:, 2] .= exp.(p[:, 2]) .* scalers[:sigma]
    return p
end

# Default inference loops for :cpu and :gpu.
# NeuroTabModelsReactantExt adds ::Val{:reactant} methods that run a Reactant-compiled forward pass.
function _infer_loop(::Val, chain, data, x0, dev, cdev, ps, st)
    preds = Vector{AbstractArray}()
    for x in data
        pred = _forward_reduce(chain, dev(x), ps, st)
        push!(preds, cdev(pred))
    end
    return preds
end

# Grouped variant: each batch is `(x, mask)`; `mask` selects real rows from padded groups.
# Architectures that mix across observations also receive `mask` as a key-padding mask.
function _infer_grp_loop(::Val, chain, data, x0, mask0, dev, cdev, ps, st)
    preds = Vector{AbstractArray}()
    use_mask = uses_batch_mask(chain)
    for (x, mask) in data
        xd = dev(x)
        xin = use_mask ? (xd, dev(mask)) : xd
        pred = _forward_reduce(chain, xin, ps, st)
        push!(preds, cdev(pred)[:, mask])
    end
    return preds
end

"""
    infer(m::NeuroTabModel, data; device=:cpu, backend=get(m.info, :backend, :zygote), proj=true)

Run inference on batched feature data.

# Arguments

- `m::NeuroTabModel`: A fitted model.
- `data`: Iterable of feature batches.

# Keyword arguments

- `device=:cpu`: Execution device (`:cpu` or `:gpu`).
- `backend`: AD backend (`:zygote`, `:enzyme`, or `:reactant`). Defaults to the value stored on the model.
- `proj=true`: When `true`, map raw outputs to natural scale. Set to `false` for raw model-scale predictions.
"""
function infer(
    m::NeuroTabModel{L}, data; device=:cpu, backend=get(m.info, :backend, :zygote), proj::Bool=true
) where {L}
    device = Symbol(device)
    backend = Symbol(backend)
    dev = _get_device(backend, device; gpuID=get(m.info, :gpuID, 0))
    cdev = cpu_device()
    ps = dev(m.info[:ps])
    st = dev(m.info[:st])
    scalers = m.info[:scalers]

    x0 = first(data)
    preds = _infer_loop(Val(backend), m.chain, data, x0, dev, cdev, ps, st)

    p_raw = _assemble(L, preds)
    proj || return p_raw
    p = _inverse_link(L, p_raw)
    return _scaler(L, p, scalers)
end

function infer_grp(
    m::NeuroTabModel{L}, data; device=:cpu, backend=get(m.info, :backend, :zygote), proj::Bool=true
) where {L}
    device = Symbol(device)
    backend = Symbol(backend)
    dev = _get_device(backend, device; gpuID=get(m.info, :gpuID, 0))
    cdev = cpu_device()
    ps = dev(m.info[:ps])
    st = dev(m.info[:st])
    scalers = m.info[:scalers]

    (x0, mask0) = first(data)
    preds = _infer_grp_loop(Val(backend), m.chain, data, x0, mask0, dev, cdev, ps, st)

    p_raw = _assemble(L, preds)
    proj || return p_raw
    p = _inverse_link(L, p_raw)
    return _scaler(L, p, scalers)
end

"""
    infer(m::NeuroTabModel, df::AbstractDataFrame; device=:cpu, backend=get(m.info, :backend, :zygote), proj=true)

Run inference on tabular data and return predictions.

# Arguments

- `m::NeuroTabModel`: A fitted model.
- `df`: Feature data as an `AbstractDataFrame`.

# Keyword arguments

- `device=:cpu`: Execution device (`:cpu` or `:gpu`).
- `backend`: AD backend (`:zygote`, `:enzyme`, or `:reactant`). Defaults to the value stored on the model.
- `proj=true`: When `true`, map raw outputs to natural scale. Set to `false` for raw model-scale outputs.
"""
function infer(
    m::NeuroTabModel, df::AbstractDataFrame; device=:cpu, backend=get(m.info, :backend, :zygote), proj::Bool=true
)
    group_key = m.info[:group_key]
    if isnothing(group_key)
        dinfer = get_df_loader_infer(df; feature_names=m.info[:feature_names], batchsize=2048)
        p = infer(m, dinfer; device, backend, proj)
    else
        dfg = groupby(df, group_key; sort=true)
        dinfer = get_df_loader_infer(dfg; feature_names=m.info[:feature_names], batchsize=2048)
        p = infer_grp(m, dinfer; device, backend, proj)
    end
    return p
end

"""
    (m::NeuroTabModel)(df::AbstractDataFrame; device=:cpu, backend=get(m.info, :backend, :zygote), proj=true)

Run inference on tabular data.
"""
function (m::NeuroTabModel)(df::AbstractDataFrame; device=:cpu, backend=get(m.info, :backend, :zygote), proj::Bool=true)
    return infer(m, df; device, backend, proj)
end

# function (m::NeuroTabModel)(x::AbstractMatrix; device=:cpu)
#     return infer(m, [(x,)]; device)
# end

end

