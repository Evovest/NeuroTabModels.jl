module Fit

export fit, fit_iter!

using ..Data
using ..Learners
using ..Models
using ..Losses
using ..Metrics

import Random: Xoshiro
import MLJModelInterface: fit
import Optimisers: OptimiserChain, WeightDecay, NAdam

using Lux
using Reactant
using Lux: cpu_device, reactant_device

using DataFrames
using CategoricalArrays

include("callback.jl")
using .CallBacks

function _get_device(config)
    backend = config.device == :gpu ? "gpu" : "cpu"
    Reactant.set_default_backend(backend)
    return reactant_device()
end

function init(
    config::LearnerTypes,
    df::AbstractDataFrame;
    feature_names,
    target_name,
    weight_name=nothing,
    offset_name=nothing
)
    dev = _get_device(config)
    batchsize = config.batchsize
    nfeats = length(feature_names)
    L = get_loss_type(config.loss)
    lux_loss = get_loss_fn(L)

    target_levels = nothing
    target_isordered = false
    outsize = 1

    if L <: MLogLoss
        eltype(df[!, target_name]) <: CategoricalValue || error("Target `$target_name` must be `<: CategoricalValue`")
        target_levels = CategoricalArrays.levels(df[!, target_name])
        target_isordered = isordered(df[!, target_name])
        outsize = length(target_levels)
    elseif L <: GaussianMLE
        outsize = 2
    end

    data = get_df_loader_train(df; feature_names, target_name, weight_name, offset_name, batchsize) |> dev

    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_levels => target_levels,
        :target_isordered => target_isordered,
        :device => config.device
    )

    chain = config.arch(; nfeats, outsize)
    m = NeuroTabModel(L, chain, info)

    rng = Xoshiro(config.seed)
    ps, st = Lux.setup(rng, m.chain) |> dev
    opt = OptimiserChain(NAdam(config.lr), WeightDecay(config.wd))
    ts = Training.TrainState(m.chain, ps, st, opt)

    return m, Dict(:data => data, :lux_loss => lux_loss, :train_state => ts)
end

"""
    function fit(
        config::LearnerTypes,
        dtrain;
        feature_names,
        target_name,
        weight_name=nothing,
        offset_name=nothing,
        deval=nothing,
        metric=nothing,
        print_every_n=9999,
        early_stopping_rounds=9999,
        verbosity=1,
    )
Training function of NeuroTabModels' internal API.
# Arguments

- `config::LearnerTypes`: The configuration object defining the model architecture, loss, and training hyperparameters.
- `dtrain`: The training data. Must be `<:AbstractDataFrame`.

# Keyword arguments

- `feature_names`: Required. A `Vector{Symbol}` or `Vector{String}` of the feature names to use.
- `target_name`: Required. A `Symbol` or `String` indicating the name of the target variable.
- `weight_name=nothing`: Optional. A `Symbol` or `String` indicating the sample weights column.
- `offset_name=nothing`: Optional. A `Symbol` or `String` indicating the offset column.
- `deval=nothing`: Optional. Evaluation data (`<:AbstractDataFrame`) for tracking metrics and early stopping.
- `metric=nothing`: Optional. The evaluation metric to track (e.g., `:mse`, `:logloss`). 
- `print_every_n=9999`: Integer. Logs training progress to the console every `N` epochs.
- `early_stopping_rounds=9999`: Integer. Stops training if the evaluation metric does not improve for this many rounds.
- `verbosity=1`: Integer. Controls the logging level (`0` for silent, `>0` for info).
- `device=:cpu`: Symbol. Hardware device to use for training (`:cpu` or `:gpu`).
- `gpuID=0`: Integer. Specifies which GPU to use if multiple are available.
"""

function fit(
    config::LearnerTypes,
    dtrain;
    feature_names,
    target_name,
    weight_name=nothing,
    offset_name=nothing,
    deval=nothing,
    print_every_n=9999,
    verbosity=1
)
    feature_names, target_name = Symbol.(feature_names), Symbol(target_name)
    weight_name = isnothing(weight_name) ? nothing : Symbol(weight_name)
    offset_name = isnothing(offset_name) ? nothing : Symbol(offset_name)

    m, cache = init(config, dtrain; feature_names, target_name, weight_name, offset_name)

    logger = nothing
    if !isnothing(deval)
        cb = CallBack(config, deval; feature_names, target_name, weight_name, offset_name)
        logger = init_logger(config)
        cb(logger, 0, cache[:train_state])
        (verbosity > 0) && @info "Init training" metric = logger[:metrics][end]
    else
        (verbosity > 0) && @info "Init training"
    end

    while m.info[:nrounds] < config.nrounds
        fit_iter!(m, cache)
        iter = m.info[:nrounds]

        if !isnothing(logger)
            cb(logger, iter, cache[:train_state])
            if verbosity > 0 && iter % print_every_n == 0
                @info "iter $iter" metric = logger[:metrics][:metric][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        else
            (verbosity > 0 && iter % print_every_n == 0) && @info "iter $iter"
        end
    end

    _sync_params_to_model!(m, cache)
    m.info[:logger] = logger
    return m
end

function _sync_params_to_model!(m, cache)
    ts = cache[:train_state]
    cdev = cpu_device()
    m.info[:ps] = cdev(ts.parameters)
    m.info[:st] = cdev(Lux.testmode(ts.states))
end

function fit_iter!(m, cache)
    ts, lux_loss = cache[:train_state], cache[:lux_loss]
    for d in cache[:data]
        _, loss, _, ts = Training.single_train_step!(AutoEnzyme(), lux_loss, d, ts)
    end
    cache[:train_state] = ts
    m.info[:nrounds] += 1
    return nothing
end

end