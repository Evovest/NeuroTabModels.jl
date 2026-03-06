module Infer

using ..Data
using ..Losses
using ..Losses: reduce_pred
using ..Models

using Lux
using Lux: cpu_device, reactant_device
using Reactant
using Reactant: @compile
using NNlib: sigmoid, softmax
using DataFrames: AbstractDataFrame
import MLUtils: DataLoader

export infer

function _get_device(device::Symbol)
    backend = device == :gpu ? "gpu" : "cpu"
    Reactant.set_default_backend(backend)
    return reactant_device()
end

_activation(::Type{<:MLogLoss}) = x -> softmax(x; dims=1)
_activation(::Type{<:LogLoss}) = x -> sigmoid.(x)
_activation(::Type{<:Tweedie}) = x -> exp.(x)
_activation(::Type) = identity

function _forward_reduce(chain, x, ps, st)
    y_pred, st_ = chain(x, ps, st)
    return reduce_pred(y_pred), st_
end

function _postprocess(::Type{<:Union{MSE,MAE}}, raw_preds)
    return vcat([vec(p) for p in raw_preds]...)
end

function _postprocess(::Type{<:LogLoss}, raw_preds)
    return vcat([vec(p) for p in raw_preds]...)
end

function _postprocess(::Type{<:MLogLoss}, raw_preds)
    p_full = reduce(hcat, raw_preds)
    return Matrix(p_full')
end

function _postprocess(::Type{<:GaussianMLE}, raw_preds)
    p_full = reduce(hcat, raw_preds)
    p_T = Matrix(p_full')
    p_T[:, 2] .= exp.(p_T[:, 2])
    return p_T
end

function _postprocess(::Type{<:Tweedie}, raw_preds)
    return vcat([vec(p) for p in raw_preds]...)
end

function infer(m::NeuroTabModel{L}, data; device=:cpu) where {L}
    dev = _get_device(device)
    cdev = cpu_device()
    ps = dev(m.info[:ps])
    st = dev(m.info[:st])
    act = _activation(L)

    raw_preds = Vector{AbstractArray}()

    b_first = first(data)
    x0 = b_first isa Tuple ? b_first[1] : b_first
    compiled = @compile _forward_reduce(m.chain, dev(x0), ps, st)

    for b in data
        x = b isa Tuple ? b[1] : b
        if size(x) == size(x0)
            y_reduced, _ = compiled(m.chain, dev(x), ps, st)
        else
            y_reduced, _ = Reactant.@jit _forward_reduce(m.chain, dev(x), ps, st)
        end
        push!(raw_preds, act(cdev(y_reduced)))
    end

    return _postprocess(L, raw_preds)
end

function infer(m::NeuroTabModel, data::AbstractDataFrame; device=:cpu)
    dinfer = get_df_loader_infer(data; feature_names=m.info[:feature_names], batchsize=2048)
    return infer(m, dinfer; device=device)
end

function (m::NeuroTabModel)(data::AbstractDataFrame; device=:cpu)
    return infer(m, data; device=device)
end

function (m::NeuroTabModel)(x::AbstractMatrix; device=:cpu)
    return infer(m, [(x,)]; device=device)
end

end