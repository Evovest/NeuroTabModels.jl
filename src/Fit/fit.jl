module Fit

export fit, fit_iter!

using ..Data
using ..Learners
using ..Models
using ..Losses
using ..Metrics
using ..Infer: reduce_pred, _get_device

import Random: Xoshiro, default_rng
import Statistics: mean, std
import MLJModelInterface: fit
import Optimisers: OptimiserChain, WeightDecay, NAdam, Adam
using Lux
using Lux: cpu_device

using DataFrames
using CategoricalArrays

get_ad_backend(backend::Symbol) = get_ad_backend(Val(backend))
get_ad_backend(::Val{:reactant}) = get_ad_backend(Val(:enzyme))
function get_ad_backend(::Val{b}) where {b}
    error(
        "Unsupported or unloaded `backend=:$b`. Supported: [:enzyme, :zygote, :reactant]. " *
        "`:enzyme` and `:reactant` require `using Enzyme`, `:zygote` requires `using Zygote`.",
    )
end

include("callback.jl")
using .CallBacks

function init(
    config::LearnerTypes,
    df::AbstractDataFrame;
    feature_names,
    target_name,
    weight_name=nothing,
    offset_name=nothing,
    group_key=nothing,
)
    feature_names, target_name = Symbol.(feature_names), Symbol(target_name)
    weight_name = isnothing(weight_name) ? nothing : Symbol(weight_name)
    offset_name = isnothing(offset_name) ? nothing : Symbol(offset_name)
    group_key = isnothing(group_key) ? nothing : Symbol(group_key)

    dev = _get_device(config.backend, config.device; gpuID=config.gpuID)
    batchsize = config.batchsize
    nfeats = length(feature_names)
    L = get_loss_type(config.loss)
    lux_loss = get_loss_fn(L)

    outsize = 1
    target_levels = nothing
    target_isordered = false

    if L <: MLogLoss
        eltype(df[!, target_name]) <: CategoricalValue || error("Target `$target_name` must be `<: CategoricalValue`")
        target_levels = CategoricalArrays.levels(df[!, target_name])
        target_isordered = isordered(df[!, target_name])
        outsize = length(target_levels)
    elseif L <: GaussianMLE
        outsize = 2
    end

    scalers = nothing
    if hasproperty(config, :scale_target) && config.scale_target && L <: Union{MSE,MAE,GaussianMLE}
        scalers = (mu=mean(df[!, target_name]), sigma=std(df[!, target_name]))
    end

    dfg = isnothing(group_key) ? df : groupby(df, group_key; sort=true)
    data = get_df_loader_train(dfg; feature_names, target_name, weight_name, offset_name, scalers, batchsize) |> dev

    # Build chain: optional embeddings + architecture backbone
    embed_config = config.embedding_config
    x_train = Models.Embeddings.needs_x_train(embed_config) ? Matrix{Float32}(df[:, feature_names]) : nothing
    embed_chain = Models.Embeddings.build_embedding_chain(embed_config, nfeats; x_train)
    ins = Models.Embeddings.embedding_width(embed_chain, randn(Float32, nfeats, 2), default_rng())
    core_chain = config.arch(; ins, outsize, loss_type=L)
    chain = Chain(embed_chain, core_chain)
    # chain = Models.build_chain(config.arch, embed_chain; ins, outsize, loss_type=L)

    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_name => target_name,
        :weight_name => weight_name,
        :offset_name => offset_name,
        :group_key => group_key,
        :target_levels => target_levels,
        :target_isordered => target_isordered,
        :scalers => scalers,
        :backend => config.backend,
        :device => config.device,
        :gpuID => config.gpuID,
    )
    m = NeuroTabModel(L, chain, info)

    rng = Xoshiro(config.seed)
    ps, st = Lux.setup(rng, m.chain) |> dev
    data = Models.train_dataloader(
        config.arch, m, data, df; feature_names, target_name, loss_type=L, scalers, batchsize, dev, rng
    )
    opt = OptimiserChain(NAdam(config.lr), WeightDecay(config.wd))
    ts = Training.TrainState(m.chain, ps, st, opt)

    return m,
    Dict(
        :data => data,
        :lux_loss => lux_loss,
        :train_state => ts,
        :scalers => scalers,
        :ad_backend => get_ad_backend(config.backend),
    )
end

"""
    fit(
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
    group_key=nothing,
    deval=nothing,
    print_every_n=9999,
    verbosity=1,
)
    m, cache = init(config, dtrain; feature_names, target_name, weight_name, offset_name, group_key)

    logger = nothing
    if !isnothing(deval)
        cb = CallBack(config, deval, cache, m; feature_names, target_name, weight_name, offset_name, group_key)
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

function _single_train_step!(device::Symbol, ad_backend, lux_loss, d, ts)
    _single_train_step!(Val(device), ad_backend, lux_loss, d, ts)
end
function _single_train_step!(::Val{D}, ad_backend, lux_loss, d, ts) where {D}
    Training.single_train_step!(ad_backend, lux_loss, d, ts)
end

function fit_iter!(m, cache)
    ts, lux_loss = cache[:train_state], cache[:lux_loss]
    ad_backend = cache[:ad_backend]
    for d in cache[:data]
        _, loss, _, ts = _single_train_step!(m.info[:backend], ad_backend, lux_loss, d, ts)
    end
    cache[:train_state] = ts
    m.info[:nrounds] += 1
    return nothing
end

end