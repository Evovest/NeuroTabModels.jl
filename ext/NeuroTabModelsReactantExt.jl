module NeuroTabModelsReactantExt

using Lux: Training, reactant_device
using NeuroTabModels
using Reactant
using Reactant: @compile

import NeuroTabModels.Infer: _get_device, _infer_loop, _infer_grp_loop
import NeuroTabModels.Fit: _single_train_step!
import NeuroTabModels.Fit.CallBacks: _compile_eval_step

using NeuroTabModels.Infer: _forward_reduce

function _get_device(::Val{:reactant}, ::Val{D}; gpuID::Integer=0) where {D}
    Reactant.set_default_backend(String(D))
    return reactant_device()
end

_same_shape(a::AbstractArray, b::AbstractArray) = size(a) == size(b)
_same_shape(a::Tuple, b::Tuple) =
    length(a) == length(b) && all(_same_shape(ai, bi) for (ai, bi) in zip(a, b))
_same_shape(_, _) = false

function _infer_loop(::Val{:reactant}, chain, data, x0, dev, cdev, ps, st)
    compiled = @compile _forward_reduce(chain, dev(x0), ps, st)

    preds = Vector{AbstractArray}()
    for x in data
        if _same_shape(x, x0)
            pred = compiled(chain, dev(x), ps, st)
        else
            pred = Reactant.@jit _forward_reduce(chain, dev(x), ps, st)
        end
        push!(preds, cdev(pred))
    end
    return preds
end

function _infer_grp_loop(::Val{:reactant}, chain, data, x0, mask0, dev, cdev, ps, st)
    use_mask = NeuroTabModels.Models.uses_batch_mask(chain)
    x0d = use_mask ? (dev(x0), dev(mask0)) : dev(x0)
    compiled = @compile _forward_reduce(chain, x0d, ps, st)

    preds = Vector{AbstractArray}()
    for (x, mask) in data
        xd = dev(x)
        xin = use_mask ? (xd, dev(mask)) : xd
        pred = compiled(chain, xin, ps, st)
        push!(preds, cdev(pred)[:, mask])
    end
    return preds
end

_single_train_step!(::Val{:reactant}, ad_backend, lux_loss, d, ts) =
    Training.single_train_step!(ad_backend, lux_loss, d, ts; return_gradients=Val(false))

_compile_eval_step(::Val{:reactant}, step, args...) = @compile step(args...)

end
