module Learners

import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema

export NeuroTabRegressor, NeuroTabClassifier, LearnerTypes

"""
    mk_rng

make a Random Number Generator object
"""
mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(rng::T) where {T<:Integer} = Random.MersenneTwister(rng)

mutable struct NeuroTabRegressor <: MMI.Deterministic
  model_type::Symbol
  loss::Symbol
  nrounds::Int
  lr::Float32
  wd::Float32
  batchsize::Int
  actA::Symbol
  depth::Int
  ntrees::Int
  hidden_size::Int
  stack_size::Int
  init_scale::Float32
  MLE_tree_split::Bool
  rng::Any
end

"""
  NeuroTabRegressor(; kwargs...)

A model type for constructing a NeuroTabRegressor, based on [NeuroTabModels.jl](https://github.com/Evovest/NeuroTabModels.jl), and implementing both an internal API and the MLJ model interface.

# Hyper-parameters

- `loss=:mse`:              Loss to be be minimized during training. One of:
  - `:mse`
  - `:mae`
  - `:logloss`
  - `:mlogloss`
  - `:gaussian_mle`
- `nrounds=100`:             Max number of rounds (epochs).
- `lr=1.0f-2`:              Learning rate. Must be > 0. A lower `eta` results in slower learning, typically requiring a higher `nrounds`.   
- `wd=0.f0`:                Weight decay applied to the gradients by the optimizer.
- `batchsize=2048`:         Batch size.
- `actA=:tanh`:             Activation function applied to each of input variable for determination of split node weight. Can be one of:
    - `:tanh`
    - `:identity`
- `depth=6`:            Depth of a tree. Must be >= 1. A tree of depth 1 has 2 prediction leaf nodes. A complete tree of depth N contains `2^N` terminal leaves and `2^N - 1` split nodes.
  Compute cost is proportional to `2^depth`. Typical optimal values are in the 3 to 5 range.
- `ntrees=64`:              Number of trees (per stack).
- `hidden_size=16`:         Size of hidden layers. Applicable only when `stack_size` > 1.
- `stack_size=1`:           Number of stacked NeuroTab blocks.
- `init_scale=1.0`:         Scaling factor applied to the predictions weights. Values in the `]0, 1]` short result in best performance. 
- `MLE_tree_split=false`:   Whether independent models are buillt for each of the 2 parameters (mu, sigma) of the the `gaussian_mle` loss.
- `rng=123`:                Either an integer used as a seed to the random number generator or an actual random number generator (`::Random.AbstractRNG`).

# Internal API

Do `config = NeuroTabRegressor()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTabRegressor(loss=:logistic, depth=5, ...)`.

## Training model

A model is trained using [`fit`](@ref):

```julia
m = fit(config, dtrain; feature_names, target_name, kwargs...)
```

## Inference

Models act as a functor. returning predictions when called as a function with features as argument:

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
config = NeuroTabRegressor(depth=5, nrounds=10)
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
m = NeuroTabRegressor(depth=5, nrounds=10)
X, y = @load_boston
mach = machine(m, X, y) |> fit!
p = predict(mach, X)
```
"""
function NeuroTabRegressor(; kwargs...)

  # defaults arguments
  args = Dict{Symbol,Any}(
    :model_type => :neurotree,
    :loss => :mse,
    :nrounds => 100,
    :lr => 1.0f-2,
    :wd => 0.0f0,
    :batchsize => 2048,
    :actA => :tanh,
    :depth => 4,
    :ntrees => 64,
    :hidden_size => 1,
    :stack_size => 1,
    :init_scale => 0.1,
    :MLE_tree_split => false,
    :rng => 123,
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

  @info "args[:model_type]" args[:model_type]
  args[:model_type] = Symbol(args[:model_type])
  @info "args[:model_type]" args[:model_type]
  args[:model_type] ∉ [:neurotree, :mlp] && error("The provided kwarg `model_type`: `$(args[:model_type])` is not supported.")

  args[:loss] = Symbol(args[:loss])
  args[:loss] ∉ [:mse, :mae, :logloss, :gaussian_mle, :tweedie_deviance] && error("The provided kwarg `loss`: `$(args[:loss])` is not supported.")

  args[:rng] = mk_rng(args[:rng])

  config = NeuroTabRegressor(
    args[:model_type],
    args[:loss],
    args[:nrounds],
    Float32(args[:lr]),
    Float32(args[:wd]),
    args[:batchsize],
    Symbol(args[:actA]),
    args[:depth],
    args[:ntrees],
    args[:hidden_size],
    args[:stack_size],
    args[:init_scale],
    args[:MLE_tree_split],
    args[:rng]
  )

  return config
end


mutable struct NeuroTabClassifier <: MMI.Probabilistic
  model_type::Symbol
  loss::Symbol
  nrounds::Int
  lr::Float32
  wd::Float32
  batchsize::Int
  actA::Symbol
  depth::Int
  ntrees::Int
  hidden_size::Int
  stack_size::Int
  init_scale::Float32
  MLE_tree_split::Bool
  rng::Any
end

"""
    NeuroTabClassifier(; kwargs...)

A model type for constructing a NeuroTabClassifier, based on [NeuroTabModels.jl](https://github.com/Evovest/NeuroTabModels.jl), and implementing both an internal API and the MLJ model interface.

# Hyper-parameters

- `nrounds=100`:             Max number of rounds (epochs).
- `lr=1.0f-2`:              Learning rate. Must be > 0. A lower `eta` results in slower learning, typically requiring a higher `nrounds`.   
- `wd=0.f0`:                Weight decay applied to the gradients by the optimizer.
- `batchsize=2048`:         Batch size.
- `actA=:tanh`:             Activation function applied to each of input variable for determination of split node weight. Can be one of:
    - `:tanh`
    - `:identity`
- `depth=6`:            Depth of a tree. Must be >= 1. A tree of depth 1 has 2 prediction leaf nodes. A complete tree of depth N contains `2^N` terminal leaves and `2^N - 1` split nodes.
  Compute cost is proportional to `2^depth`. Typical optimal values are in the 3 to 5 range.
- `ntrees=64`:              Number of trees (per stack).
- `hidden_size=16`:         Size of hidden layers. Applicable only when `stack_size` > 1.
- `stack_size=1`:           Number of stacked NeuroTab blocks.
- `init_scale=1.0`:         Scaling factor applied to the predictions weights. Values in the `]0, 1]` short result in best performance. 
- `MLE_tree_split=false`:   Whether independent models are buillt for each of the 2 parameters (mu, sigma) of the the `gaussian_mle` loss.
- `rng=123`:                Either an integer used as a seed to the random number generator or an actual random number generator (`::Random.AbstractRNG`).

# Internal API

Do `config = NeuroTabClassifier()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTabClassifier(depth=5, ...)`.

## Training model

A model is trained using [`fit`](@ref):

```julia
m = fit(config, dtrain; feature_names, target_name, kwargs...)
```

## Inference

Models act as a functor. returning predictions when called as a function with features as argument:

```julia
m(data)
```

# MLJ Interface

From MLJ, the type can be imported using:

```julia
NeuroTabClassifier = @load NeuroTabClassifier pkg=NeuroTabModels
```

Do `model = NeuroTabClassifier()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTabClassifier(loss=...)`.

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
config = NeuroTabClassifier(depth=5, nrounds=10)
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
m = NeuroTabClassifier(depth=5, nrounds=10)
X, y = @load_crabs
mach = machine(m, X, y) |> fit!
p = predict(mach, X)
```
"""
function NeuroTabClassifier(; kwargs...)

  # defaults arguments
  args = Dict{Symbol,Any}(
    :model_type => :neurotree,
    :nrounds => 100,
    :lr => 1.0f-2,
    :wd => 0.0f0,
    :batchsize => 2048,
    :actA => :tanh,
    :depth => 4,
    :ntrees => 64,
    :hidden_size => 1,
    :stack_size => 1,
    :init_scale => 0.1,
    :MLE_tree_split => false,
    :rng => 123,
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

  args[:model_type] = Symbol(args[:model_type])
  args[:model_type] ∉ [:neurotree, :mlp] && error("The provided kwarg `model_type`: `$model_type` is not supported.")

  args[:rng] = mk_rng(args[:rng])

  config = NeuroTabClassifier(
    args[:model_type],
    :mlogloss,
    args[:nrounds],
    Float32(args[:lr]),
    Float32(args[:wd]),
    args[:batchsize],
    Symbol(args[:actA]),
    args[:depth],
    args[:ntrees],
    args[:hidden_size],
    args[:stack_size],
    args[:init_scale],
    args[:MLE_tree_split],
    args[:rng],
  )

  return config
end

const LearnerTypes = Union{NeuroTabRegressor,NeuroTabClassifier}

end