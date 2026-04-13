module Infer

using ..Data
using ..Losses
using ..Models

using Lux
using Lux: cpu_device, reactant_device
using Reactant
using Reactant: @compile
using NNlib: sigmoid, softmax
using Statistics: mean
using DataFrames: AbstractDataFrame
import MLUtils: DataLoader

export infer, reduce_pred

reduce_pred(pred::AbstractMatrix) = pred
reduce_pred(pred::AbstractArray{T,3}) where {T} = dropdims(mean(pred; dims=2); dims=2)

function _get_device(device::Symbol)
    backend = device == :gpu ? "gpu" : "cpu"
    Reactant.set_default_backend(backend)
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
_inverse_link(::Type{<:LogLoss}, pred, scalers) = sigmoid.(pred)
_inverse_link(::Type{<:Tweedie}, pred, scalers) = exp.(pred)
_inverse_link(::Type{<:Union{MSE,MAE}}, pred, scalers) = pred .* scalers[:sigma] .+ scalers[:mu]

function _inverse_link(::Type{<:MLogLoss}, pred, scalers)
    return Matrix(softmax(pred; dims=1)')
end

function _inverse_link(::Type{<:GaussianMLE}, pred, scalers)
    p = Matrix(pred')
    @views p[:, 1] .= p[:, 1] .* scalers[:sigma] .+ scalers[:mu]
    @views p[:, 2] .= exp.(p[:, 2]) .* scalers[:sigma]
    return p
end

function _postprocess(::Type{L}, preds; proj::Bool=true, scalers=nothing) where {L}
    p_raw = _assemble(L, preds)
    proj || return p_raw
    return _inverse_link(L, p_raw, scalers)
end

function infer(m::NeuroTabModel{L}, data; device=:cpu, proj::Bool=false) where {L}
    dev = _get_device(device)
    cdev = cpu_device()
    ps = dev(m.info[:ps])
    st = dev(m.info[:st])

    preds = Vector{AbstractArray}()

    b_first = first(data)
    x0 = b_first isa Tuple ? b_first[1] : b_first
    compiled = @compile _forward_reduce(m.chain, dev(x0), ps, st)

    for b in data
        x = b isa Tuple ? b[1] : b
        if size(x) == size(x0)
            pred = compiled(m.chain, dev(x), ps, st)
        else
            pred = Reactant.@jit _forward_reduce(m.chain, dev(x), ps, st)
        end
        push!(preds, cdev(pred))
    end
    return _postprocess(L, preds; proj, scalers=m.info[:scalers])
end

function infer(m::NeuroTabModel, data::AbstractDataFrame; device=:cpu, proj::Bool=true)
    dinfer = get_df_loader_infer(data; feature_names=m.info[:feature_names], batchsize=2048)
    return infer(m, dinfer; device, proj)
end

function (m::NeuroTabModel)(data::AbstractDataFrame; device=:cpu, proj::Bool=true)
    return infer(m, data; device, proj)
end

end