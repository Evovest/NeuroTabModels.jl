module CallBacks

using DataFrames
using Statistics: mean, median

using ..Learners: LearnerTypes
using ..Losses: reduce_pred
using ..Data: get_df_loader_train
using ..Metrics

using Lux: Training, reactant_device, testmode
using Reactant: @compile

export CallBack, init_logger, update_logger!, agg_logger

struct CallBack{D,C}
    deval::D
    eval_compiled::C
end

function (cb::CallBack)(logger, iter, ts::Training.TrainState)
    metric = Metrics.get_metric(ts, cb.deval, cb.eval_compiled)
    update_logger!(logger; iter, metric)
    return nothing
end

function CallBack(
    config::LearnerTypes,
    deval::AbstractDataFrame,
    ts::Training.TrainState;
    feature_names,
    target_name,
    weight_name=nothing,
    offset_name=nothing
)
    dev = reactant_device()
    batchsize = config.batchsize
    feval = metric_dict[config.metric]
    deval = get_df_loader_train(deval; feature_names, target_name, weight_name, offset_name, batchsize, shuffle=false) |> dev

    ps, st = ts.parameters, testmode(ts.states)
    d0 = first(deval)
    eval_compiled = _compile_eval_step(ts.model, feval, d0, ps, st)

    return CallBack(deval, eval_compiled)
end

function _compile_eval_step(chain, feval, d0, ps, st)
    if length(d0) == 2
        function _step2(x, y, ps, st)
            m = x -> reduce_pred(first(chain(x, ps, st)))
            return feval(m, x, y; agg=sum), eltype(y)(last(size(y)))
        end
        return @compile _step2(d0[1], d0[2], ps, st)
    elseif length(d0) == 3
        function _step3(x, y, w, ps, st)
            m = x -> reduce_pred(first(chain(x, ps, st)))
            return feval(m, x, y, w; agg=sum), sum(w)
        end
        return @compile _step3(d0[1], d0[2], d0[3], ps, st)
    else
        function _step4(x, y, w, offset, ps, st)
            m = x -> reduce_pred(first(chain(x, ps, st)))
            return feval(m, x, y, w, offset; agg=sum), sum(w)
        end
        return @compile _step4(d0[1], d0[2], d0[3], d0[4], ps, st)
    end
end

function init_logger(config::LearnerTypes)
    logger = Dict(
        :name => String(config.metric),
        :maximise => is_maximise(metric_dict[config.metric]),
        :early_stopping_rounds => config.early_stopping_rounds,
        :nrounds => 0,
        :metrics => (iter=Int[], metric=Float64[]),
        :iter_since_best => 0,
        :best_iter => 0,
        :best_metric => 0.0,
    )
    return logger
end

function update_logger!(logger; iter, metric)
    logger[:nrounds] = iter
    push!(logger[:metrics][:iter], iter)
    push!(logger[:metrics][:metric], metric)
    if iter == 0
        logger[:best_metric] = metric
    else
        if (logger[:maximise] && metric > logger[:best_metric]) ||
           (!logger[:maximise] && metric < logger[:best_metric])
            logger[:best_metric] = metric
            logger[:best_iter] = iter
            logger[:iter_since_best] = 0
        else
            logger[:iter_since_best] += logger[:metrics][:iter][end] - logger[:metrics][:iter][end-1]
        end
    end
end

function agg_logger(logger_raw::Vector{Dict})
    _l1 = first(logger_raw)
    best_iters = [d[:best_iter] for d in logger_raw]
    best_iter = ceil(Int, median(best_iters))

    best_metrics = [d[:best_metric] for d in logger_raw]
    best_metric = last(best_metrics)

    metrics = (layer=Int[], iter=Int[], metric=Float64[])
    for i in eachindex(logger_raw)
        _l = logger_raw[i]
        append!(metrics[:layer], zeros(Int, length(_l[:metrics][:iter])) .+ i)
        append!(metrics[:iter], _l[:metrics][:iter])
        append!(metrics[:metric], _l[:metrics][:metric])
    end

    logger = Dict(
        :name => _l1[:name],
        :maximise => _l1[:maximise],
        :early_stopping_rounds => _l1[:early_stopping_rounds],
        :metrics => metrics,
        :best_iters => best_iters,
        :best_iter => best_iter,
        :best_metrics => best_metrics,
        :best_metric => best_metric,
    )
    return logger
end

end