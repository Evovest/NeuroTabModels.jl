module Infer

using ..Data
using ..Losses
using ..Models
using ..Learners

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

function _postprocess(::Type{<:Union{MSE,MAE}}, raw_preds)
    return vcat([vec(p) for p in raw_preds]...)
end

function _postprocess(::Type{<:LogLoss}, raw_preds)
    p = vcat([vec(p) for p in raw_preds]...)
    return sigmoid.(p)
end

function _postprocess(::Type{<:MLogLoss}, raw_preds)
    p_full = reduce(hcat, raw_preds)
    p_soft = softmax(p_full; dims=1)
    return Matrix(p_soft')
end

function _postprocess(::Type{<:GaussianMLE}, raw_preds)
    p_full = reduce(hcat, raw_preds)
    p_T = Matrix(p_full')
    p_T[:, 2] .= exp.(p_T[:, 2])
    return p_T
end

function _postprocess(::Type{<:Tweedie}, raw_preds)
    p = vcat([vec(p) for p in raw_preds]...)
    return exp.(p)
end

function infer(m::NeuroTabModel{L}, data; device=:cpu) where {L}
    dev = _get_device(device)
    cdev = cpu_device()
    ps = dev(m.info[:ps])
    st = dev(m.info[:st])

    raw_preds = Vector{AbstractArray}()

    b_first = first(data)
    x0 = b_first isa Tuple ? b_first[1] : b_first
    model_compiled = @compile m.chain(dev(x0), ps, st)

    for b in data
        x = b isa Tuple ? b[1] : b
        if size(x) == size(x0)
            y_pred, _ = model_compiled(dev(x), ps, st)
        else
            y_pred, _ = Reactant.@jit Lux.apply(m.chain, dev(x), ps, st)
        end
        push!(raw_preds, cdev(y_pred))
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