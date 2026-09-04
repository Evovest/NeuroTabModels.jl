module Learners

import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema
using Random: Random

using ..Models
using ..Models: AbstractEmbedding, IdentityEmbedding, EmbeddingLayer
export NeuroTabRegressor, NeuroTabClassifier, LearnerTypes

_to_embedding(::Nothing) = IdentityEmbedding()
_to_embedding(e::AbstractEmbedding) = e
_to_embedding(d::AbstractDict) = EmbeddingLayer(d)
function _to_embedding(x)
    error("`embedding_config` must be `nothing`, an `AbstractDict`, or an `AbstractEmbedding`; got $(typeof(x)).")
end

mutable struct NeuroTabRegressor <: MMI.Deterministic
    loss::Symbol
    metric::Symbol
    arch::Architecture
    embedding_config::AbstractEmbedding
    nrounds::Int
    early_stopping_rounds::Int
    lr::Float32
    wd::Float32
    batchsize::Int
    seed::Int
    scale_target::Bool
    backend::Symbol
    device::Symbol
    gpuID::Int
end

"""
  NeuroTabRegressor(arch::Architecture; kwargs...)
  NeuroTabRegressor(; arch_name="NeuroTreeConfig", arch_config::AbstractDict=Dict(), kwargs...)

A model type for constructing a NeuroTabRegressor, based on [NeuroTabModels.jl](https://github.com/Evovest/NeuroTabModels.jl), and implementing both an internal API and the MLJ model interface.

# Hyper-parameters

- `loss=:mse`:              Loss to be minimized during training. One of:
  - `:mse`
  - `:mae`
  - `:logloss`
  - `:tweedie`
  - `:gaussian_mle`
  - `:correlation`
- `nrounds=10`:             Max number of rounds (epochs).
- `lr=1.0f-2`:              Learning rate. Must be > 0. A lower `lr` results in slower learning, typically requiring a higher `nrounds`.
- `wd=0.f0`:                Weight decay applied to the gradients by the optimizer.
- `batchsize=2048`:         Batch size.
- `seed=123`:               An integer used as a seed to the random number generator.
- `backend=:zygote`:        Backend used by Lux. One of `:enzyme`, `:zygote`, or `:reactant`.
- `device=:gpu`:            Execution device. One of `:cpu` or `:gpu`.
- `gpuID=0`:                GPU device to use, only relevant if `device = :gpu`. `0` auto-selects.
- `embedding_config=nothing`: Optional numerical/temporal embeddings. Accepts `nothing` (no-op),
  an `AbstractEmbedding` (e.g. `EmbeddingLayer(num=PeriodicEmbeddings(d_embedding=24))`), or an
  `AbstractDict` selecting the type via `:embedding_type` (e.g.
  `Dict(:embedding_type => :periodic, :d_embedding => 24)`).

`backend=:zygote` works on `:cpu` and `:gpu`; `backend=:reactant` works on `:cpu` and `:gpu` and uses Enzyme for AD.
`backend=:enzyme` with `device=:gpu` is currently known to fail for some NeuroTabModels models.

# Internal API

Do `config = NeuroTabRegressor()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTabRegressor(loss=:mse, nrounds=10, ...)`.

## Training model

A model is trained using [`fit`](@ref):

```julia
m = fit(config, dtrain; feature_names, target_name, kwargs...)
```

## Inference

Models act as a functor, returning predictions when called as a function with features as argument:

```julia
m(data)
```

# MLJ Interface

From MLJ, the type can be imported using:

```julia
NeuroTabRegressor = @load NeuroTabRegressor pkg=NeuroTabModels
```

Do `model = NeuroTabRegressor()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTabRegressor(loss=...)`.

## Training model

In MLJ or MLJBase, bind an instance `model` to data with
    `mach = machine(model, X, y)` where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:Continuous`; check the scitype
  with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.

## Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above.

## Fitted parameters

The fields of `fitted_params(mach)` are:
  - `:fitresult`: The `NeuroTabModel` object.

## Report

The fields of `report(mach)` are:
  - `:features`: The names of the features encountered in training.

# Examples

## Internal API

```julia
using NeuroTabModels, DataFrames
config = NeuroTabRegressor(NeuroTreeConfig(; depth=5); nrounds=10)
nobs, nfeats = 1_000, 5
dtrain = DataFrame(randn(nobs, nfeats), :auto)
dtrain.y = rand(nobs)
feature_names, target_name = names(dtrain, r"x"), "y"
m = fit(config, dtrain; feature_names, target_name)
p = m(dtrain)
```

## MLJ Interface

```julia
using MLJBase, NeuroTabModels
m = NeuroTabRegressor(NeuroTreeConfig(; depth=5); nrounds=10)
X, y = @load_boston
mach = machine(m, X, y) |> fit!
p = predict(mach, X)
```
"""
function NeuroTabRegressor(arch::Architecture; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :loss => :mse,
        :metric => nothing,
        :nrounds => 10,
        :early_stopping_rounds => typemax(Int),
        :lr => 1.0f-2,
        :wd => 0.0f0,
        :batchsize => 2048,
        :seed => 123,
        :backend => :zygote,
        :device => :gpu,
        :gpuID => 0,
        :embedding_config => nothing,
        :scale_target => true,
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    args_ignored_str = join(args_ignored, ", ")
    length(args_ignored) > 0 &&
        @info "Following $(length(args_ignored)) provided arguments will be ignored: $(args_ignored_str)."

    args_default = setdiff(keys(args), keys(kwargs))
    args_default_str = join(args_default, ", ")
    length(args_default) > 0 &&
        @info "Following $(length(args_default)) arguments were not provided and will be set to default: $(args_default_str)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    loss = Symbol(args[:loss])
    loss ∉ [:mse, :mae, :logloss, :tweedie, :gaussian_mle, :correlation] &&
        error("The provided kwarg `loss`: $loss is not supported.")

    _metric_list = [:mse, :mae, :logloss, :tweedie, :gaussian_mle, :correlation]
    if isnothing(args[:metric])
        metric = loss
    else
        metric = Symbol(args[:metric])
    end
    if metric ∉ _metric_list
        error("Invalid metric. Must be one of: $_metric_list")
    end

    backend = Symbol(args[:backend])
    device = Symbol(args[:device])
    if device == :reactant
        error("Use `backend=:reactant` with `device=:cpu` or `device=:gpu` instead of `device=:reactant`.")
    end
    if backend == :enzyme && device == :gpu
        @warn "`backend=:enzyme` with `device=:gpu` is currently known to fail for some NeuroTabModels models. Prefer `backend=:zygote` on GPU, `backend=:reactant` for Reactant, or `backend=:enzyme` on CPU."
    end

    embed = _to_embedding(args[:embedding_config])

    config = NeuroTabRegressor(
        loss,
        metric,
        arch,
        embed,
        args[:nrounds],
        args[:early_stopping_rounds],
        Float32(args[:lr]),
        Float32(args[:wd]),
        args[:batchsize],
        args[:seed],
        args[:scale_target],
        backend,
        device,
        args[:gpuID],
    )

    return config
end

function NeuroTabRegressor(; arch_name="NeuroTreeConfig", arch_config::AbstractDict=Dict(), kwargs...)
    arch_type = eval(Meta.parse(arch_name))
    arch = arch_type(; arch_config...)
    return NeuroTabRegressor(arch; kwargs...)
end

mutable struct NeuroTabClassifier <: MMI.Probabilistic
    loss::Symbol
    metric::Symbol
    arch::Architecture
    embedding_config::AbstractEmbedding
    nrounds::Int
    early_stopping_rounds::Int
    lr::Float32
    wd::Float32
    batchsize::Int
    seed::Int
    backend::Symbol
    device::Symbol
    gpuID::Int
end

"""
  NeuroTabClassifier(arch::Architecture; kwargs...)
  NeuroTabClassifier(; arch_name="NeuroTreeConfig", arch_config::AbstractDict=Dict(), kwargs...)

A model type for constructing a NeuroTabClassifier, based on [NeuroTabModels.jl](https://github.com/Evovest/NeuroTabModels.jl), and implementing both an internal API and the MLJ model interface.

# Hyper-parameters

- `nrounds=10`:             Max number of rounds (epochs).
- `lr=1.0f-2`:              Learning rate. Must be > 0. A lower `lr` results in slower learning, typically requiring a higher `nrounds`.
- `wd=0.f0`:                Weight decay applied to the gradients by the optimizer.
- `batchsize=2048`:         Batch size.
- `seed=123`:               An integer used as a seed to the random number generator.
- `backend=:zygote`:        Backend used by Lux. One of `:enzyme`, `:zygote`, or `:reactant`.
- `device=:gpu`:            Execution device. One of `:cpu` or `:gpu`.
- `gpuID=0`:                GPU device to use, only relevant if `device = :gpu`. `0` auto-selects.
- `embedding_config=nothing`: Optional numerical/temporal embeddings. Accepts `nothing` (no-op),
  an `AbstractEmbedding`, or an `AbstractDict` selecting the type via `:embedding_type`.

`backend=:zygote` works on `:cpu` and `:gpu`; `backend=:reactant` works on `:cpu` and `:gpu` and uses Enzyme for AD.
`backend=:enzyme` with `device=:gpu` is currently known to fail for some NeuroTabModels models.

# Internal API

Do `config = NeuroTabClassifier()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTabClassifier(nrounds=10, ...)`.

## Training model

A model is trained using [`fit`](@ref):

```julia
m = fit(config, dtrain; feature_names, target_name, kwargs...)
```

## Inference

Models act as a functor, returning predictions when called as a function with features as argument:

```julia
m(data)
```

# MLJ Interface

From MLJ, the type can be imported using:

```julia
NeuroTabClassifier = @load NeuroTabClassifier pkg=NeuroTabModels
```

Do `model = NeuroTabClassifier()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTabClassifier(nrounds=...)`.

## Training model

In MLJ or MLJBase, bind an instance `model` to data with
    `mach = machine(model, X, y)` where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:Finite`; check the scitype
  with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.

## Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above.

## Fitted parameters

The fields of `fitted_params(mach)` are:
  - `:fitresult`: The `NeuroTabModel` object.

## Report

The fields of `report(mach)` are:
  - `:features`: The names of the features encountered in training.

# Examples

## Internal API

```julia
using NeuroTabModels, DataFrames, CategoricalArrays, Random
config = NeuroTabClassifier(NeuroTreeConfig(; depth=5); nrounds=10)
nobs, nfeats = 1_000, 5
dtrain = DataFrame(randn(nobs, nfeats), :auto)
dtrain.y = categorical(rand(1:2, nobs))
feature_names, target_name = names(dtrain, r"x"), "y"
m = fit(config, dtrain; feature_names, target_name)
p = m(dtrain)
```

## MLJ Interface

```julia
using MLJBase, NeuroTabModels
m = NeuroTabClassifier(NeuroTreeConfig(; depth=5); nrounds=10)
X, y = @load_crabs
mach = machine(m, X, y) |> fit!
p = predict(mach, X)
```
"""
function NeuroTabClassifier(arch::Architecture; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :metric => nothing,
        :nrounds => 10,
        :early_stopping_rounds => typemax(Int),
        :lr => 1.0f-2,
        :wd => 0.0f0,
        :batchsize => 2048,
        :seed => 123,
        :backend => :zygote,
        :device => :gpu,
        :gpuID => 0,
        :embedding_config => nothing,
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    args_ignored_str = join(args_ignored, ", ")
    length(args_ignored) > 0 &&
        @info "Following $(length(args_ignored)) provided arguments will be ignored: $(args_ignored_str)."

    args_default = setdiff(keys(args), keys(kwargs))
    args_default_str = join(args_default, ", ")
    length(args_default) > 0 &&
        @info "Following $(length(args_default)) arguments were not provided and will be set to default: $(args_default_str)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    backend = Symbol(args[:backend])
    device = Symbol(args[:device])
    if device == :reactant
        error("Use `backend=:reactant` with `device=:cpu` or `device=:gpu` instead of `device=:reactant`.")
    end
    if backend == :enzyme && device == :gpu
        @warn "`backend=:enzyme` with `device=:gpu` is currently known to fail for some NeuroTabModels models. Prefer `backend=:zygote` on GPU, `backend=:reactant` for Reactant, or `backend=:enzyme` on CPU."
    end

    embed = _to_embedding(args[:embedding_config])

    config = NeuroTabClassifier(
        :mlogloss,
        :mlogloss,
        arch,
        embed,
        args[:nrounds],
        args[:early_stopping_rounds],
        Float32(args[:lr]),
        Float32(args[:wd]),
        args[:batchsize],
        args[:seed],
        backend,
        device,
        args[:gpuID],
    )

    return config
end

function NeuroTabClassifier(; arch_name="NeuroTreeConfig", arch_config::AbstractDict=Dict(), kwargs...)
    arch_type = eval(Meta.parse(arch_name))
    arch = arch_type(; arch_config...)
    return NeuroTabClassifier(arch; kwargs...)
end

const LearnerTypes = Union{NeuroTabRegressor,NeuroTabClassifier}

end