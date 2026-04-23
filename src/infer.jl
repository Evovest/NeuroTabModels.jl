module Infer

using ..Data
using ..Losses
using ..Models

using Lux
using Lux: cpu_device, gpu_device, reactant_device
using Reactant
using Reactant: @compile
using NNlib: sigmoid, softmax
using Statistics: mean
using DataFrames: AbstractDataFrame, GroupedDataFrame, groupby
import MLUtils: DataLoader

export infer, reduce_pred

reduce_pred(pred::AbstractMatrix) = pred
reduce_pred(pred::AbstractArray{T,3}) where {T} = dropdims(mean(pred; dims=2); dims=2)


"""
    _get_device(device::Symbol)

!!! warning
    Returns the default reactant device. 
    Future behavior should support :cpu/gpu device in addition to the assumed :reactant device.
"""
function _get_device(device::Symbol)
    return reactant_device()
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
_inverse_link(::Type{<:Union{MSE,MAE}}, pred) = pred
_inverse_link(::Type{<:MLogLoss}, pred) = Matrix(softmax(pred; dims=1)')
function _inverse_link(::Type{<:GaussianMLE}, pred)
    p = Matrix(pred')
    @views p[:, 1] .= p[:, 1]
    @views p[:, 2] .= exp.(p[:, 2])
    return p
end

_scaler(::Type{<:LossType}, p, scalers) = p
_scaler(::Type{<:Union{MSE,MAE}}, p, scalers::NamedTuple) = p .* scalers[:sigma] .+ scalers[:mu]
function _scaler(::Type{<:GaussianMLE}, p, scalers::NamedTuple)
    @views p[:, 1] .= p[:, 1] .* scalers[:sigma] .+ scalers[:mu]
    @views p[:, 2] .= exp.(p[:, 2]) .* scalers[:sigma]
    return p
end

function infer(m::NeuroTabModel{L}, data; device=:cpu, proj::Bool=true) where {L}
    dev = _get_device(Symbol(device))
    cdev = cpu_device()
    ps = dev(m.info[:ps])
    st = dev(m.info[:st])
    scalers = m.info[:scalers]

    x0 = first(data)
    compiled = @compile _forward_reduce(m.chain, dev(x0), ps, st)

    preds = Vector{AbstractArray}()
    for x in data
        if size(x) == size(x0)
            pred = compiled(m.chain, dev(x), ps, st)
        else
            pred = Reactant.@jit _forward_reduce(m.chain, dev(x), ps, st)
        end
        push!(preds, cdev(pred))
    end

    p_raw = _assemble(L, preds)
    proj || return p_raw
    p = _inverse_link(L, p_raw)
    return _scaler(L, p, scalers)
end

function infer_grp(m::NeuroTabModel{L}, data; device=:cpu, proj::Bool=true) where {L}
    dev = _get_device(Symbol(device))
    cdev = cpu_device()
    ps = dev(m.info[:ps])
    st = dev(m.info[:st])
    scalers = m.info[:scalers]

    (x0, mask0) = first(data)
    # @info typeof("mask0") mask0
    # data = data |> dev
    # (x0, mask0) = first(data)
    # @info typeof("mask0-dev") mask0
    compiled = @compile _forward_reduce(m.chain, dev(x0), ps, st)

    preds = Vector{AbstractArray}()
    for (x, mask) in data
        pred = compiled(m.chain, dev(x), ps, st)
        push!(preds, cdev(pred)[:, mask])
    end

    p_raw = _assemble(L, preds)
    proj || return p_raw
    p = _inverse_link(L, p_raw)
    return _scaler(L, p, scalers)
end

function infer(m::NeuroTabModel, df::AbstractDataFrame; device=:cpu, proj::Bool=true)
    group_key = m.info[:group_key]
    if isnothing(group_key)
        dinfer = get_df_loader_infer(df; feature_names=m.info[:feature_names], batchsize=2048)
        p = infer(m, dinfer; device, proj)
    else
        dfg = groupby(df, group_key; sort=true)
        dinfer = get_df_loader_infer(dfg; feature_names=m.info[:feature_names], batchsize=2048)
        p = infer_grp(m, dinfer; device, proj)
    end
    return p
end

function (m::NeuroTabModel)(df::AbstractDataFrame; device=:cpu, proj::Bool=true)
    return infer(m, df; device, proj)
end

# function (m::NeuroTabModel)(x::AbstractMatrix; device=:cpu)
#     return infer(m, [(x,)]; device)
# end

end