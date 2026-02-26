# Getting started with NeuroTabModels.jl

## Installation

```julia
] add NeuroTabModels
```

## Configuring a model

A model configuration is defined with the [NeuroTabRegressor](@ref) constructor:

```julia
using NeuroTabModels, DataFrames
using NeuroTabModels.Models

arch = NeuroTreeConfig(depth=4, ntrees=16);
config = NeuroTabRegressor(
    arch;
    loss = :mse,
    nrounds = 10,
    depth = 5,
)

# alternative kwarg-only syntax
config = NeuroTabRegressor(;
    arch_name = "NeuroTreeConfig",
    arch_config = Dict(:depth => 4, :ntrees => 16),
    loss = :mse,
    nrounds = 10,
)
```

## Training

Building and training a model according to the above `config` is done with [NeuroTabModels.fit](@ref).
See the docs for additional features, notably early stopping support through the tracking of an evaluation metric.

```julia
nobs, nfeats = 1_000, 5
dtrain = DataFrame(randn(nobs, nfeats), :auto)
dtrain.y = rand(nobs)
feature_names, target_name = names(dtrain, r"x"), "y"

m = NeuroTabModels.fit(config, dtrain; feature_names, target_name)
```

## Inference

```julia
p = m(dtrain)
```

## MLJ

NeuroTabModels.jl supports the [MLJ](https://github.com/alan-turing-institute/MLJ.jl) Interface. 

```julia
using MLJBase, NeuroTabModels
m = NeuroTabRegressor(depth=4, nrounds=10)
X, y = @load_boston
mach = machine(m, X, y) |> fit!
p = predict(mach, X)
```

## Benchmarks

Benchmarking against prominent ML algos for tabular data is performed at [MLBenchmarks.jl](https://github.com/Evovest/MLBenchmarks.jl).
