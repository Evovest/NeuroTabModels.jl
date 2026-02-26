# NeuroTabModels.jl

> Differentiable models for tabular data. 

| Documentation | CI Status | DOI |
|:------------------------:|:----------------:|:----------------:|
| [![][docs-latest-img]][docs-latest-url] | [![][ci-img]][ci-url] | [![][DOI-img]][DOI-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://evovest.github.io/NeuroTabModels.jl/dev

[ci-img]: https://github.com/Evovest/NeuroTabModels.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/Evovest/NeuroTabModels.jl/actions?query=workflow%3ACI+branch%3Amain

[DOI-img]: https://zenodo.org/badge/762536508.svg
[DOI-url]: https://zenodo.org/doi/10.5281/zenodo.10725028

## Installation

```julia
] add NeuroTabModels
```

⚠ Compatible with Julia >= v1.9.

## Configuring a model

A model configuration is defined with on of the constructor:
- [NeuroTabRegressor](https://evovest.github.io/NeuroTabModels.jl/dev/models#NeuroTabModels.NeuroTabRegressor)
- [NeuroTabClassifier](https://evovest.github.io/NeuroTabModels.jl/dev/models#NeuroTabModels.NeuroTabClassifier)

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

Building and training a model according to the above `config` is done with [NeuroTabModels.fit](https://evovest.github.io/NeuroTabModels.jl/dev/API#NeuroTabModels.fit).
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
