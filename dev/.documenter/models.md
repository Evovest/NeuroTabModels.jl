
# Models {#Models}

## NeuroTabRegressor {#NeuroTabRegressor}
<details class='jldocstring custom-block' open>
<summary><a id='NeuroTabModels.Learners.NeuroTabRegressor' href='#NeuroTabModels.Learners.NeuroTabRegressor'><span class="jlbinding">NeuroTabModels.Learners.NeuroTabRegressor</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



NeuroTabRegressor(arch::Architecture; kwargs...)   NeuroTabRegressor(; arch_name=&quot;NeuroTreeConfig&quot;, arch_config::AbstractDict=Dict(), kwargs...)

A model type for constructing a NeuroTabRegressor, based on [NeuroTabModels.jl](https://github.com/Evovest/NeuroTabModels.jl), and implementing both an internal API and the MLJ model interface.

**Hyper-parameters**
- `loss=:mse`:              Loss to be be minimized during training. One of:
  - `:mse`
    
  - `:mae`
    
  - `:logloss`
    
  - `:mlogloss`
    
  - `:gaussian_mle`
    
  
- `nrounds=100`:             Max number of rounds (epochs).
  
- `lr=1.0f-2`:              Learning rate. Must be &gt; 0. A lower `eta` results in slower learning, typically requiring a higher `nrounds`.   
  
- `wd=0.f0`:                Weight decay applied to the gradients by the optimizer.
  
- `batchsize=2048`:         Batch size.
  
- `rng=123`:                Either an integer used as a seed to the random number generator or an actual random number generator (`::Random.AbstractRNG`).
  
- `device=:cpu`:            Device on which to perform the computation, either `:cpu` or `:gpu`
  
- `gpuID=0`:                GPU device to use, only relveant if `device = :gpu` 
  

**Internal API**

Do `config = NeuroTabRegressor()` to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTabRegressor(loss=:logistic, depth=5, ...)`.

**Training model**

A model is trained using [`fit`](/API#MLJModelInterface.fit):

```julia
m = fit(config, dtrain; feature_names, target_name, kwargs...)
```


**Inference**

Models act as a functor. returning predictions when called as a function with features as argument:

```julia
m(data)
```


**MLJ Interface**

From MLJ, the type can be imported using:

```julia
NeuroTabRegressor = @load NeuroTabRegressor pkg=NeuroTabModels
```


Do `model = NeuroTabRegressor()` to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTabRegressor(loss=...)`.

**Training model**

In MLJ or MLJBase, bind an instance `model` to data with     `mach = machine(model, X, y)` where
- `X`: any table of input features (eg, a `DataFrame`) whose columns each have one of the following element scitypes: `Continuous`, `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
  
- `y`: is the target, which can be any `AbstractVector` whose element scitype is `<:Continuous`; check the scitype with `scitype(y)`
  

Train the machine using `fit!(mach, rows=...)`.

**Operations**
- `predict(mach, Xnew)`: return predictions of the target given features `Xnew` having the same scitype as `X` above.
  

**Fitted parameters**

The fields of `fitted_params(mach)` are:
- `:fitresult`: The `NeuroTabModel` object.
  

**Report**

The fields of `report(mach)` are:
- `:features`: The names of the features encountered in training.
  

**Examples**

**Internal API**

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


**MLJ Interface**

```julia
using MLJBase, NeuroTabModels
m = NeuroTabRegressor(depth=5, nrounds=10)
X, y = @load_boston
mach = machine(m, X, y) |> fit!
p = predict(mach, X)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/Evovest/NeuroTabModels.jl/blob/cc9c3677afe6b703742d9667d6cfda07ba24784b/src/learners.jl#L32-L138" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## NeuroTabClassifier {#NeuroTabClassifier}
<details class='jldocstring custom-block' open>
<summary><a id='NeuroTabModels.Learners.NeuroTabClassifier' href='#NeuroTabModels.Learners.NeuroTabClassifier'><span class="jlbinding">NeuroTabModels.Learners.NeuroTabClassifier</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



NeuroTabClassifier(arch::Architecture; kwargs...)   NeuroTabClassifier(; arch_name=&quot;NeuroTreeConfig&quot;, arch_config::AbstractDict=Dict(), kwargs...)

A model type for constructing a NeuroTabClassifier, based on [NeuroTabModels.jl](https://github.com/Evovest/NeuroTabModels.jl), and implementing both an internal API and the MLJ model interface.

**Hyper-parameters**
- `loss=:mse`:              Loss to be be minimized during training. One of:
  - `:mse`
    
  - `:mae`
    
  - `:logloss`
    
  - `:mlogloss`
    
  - `:gaussian_mle`
    
  
- `nrounds=100`:             Max number of rounds (epochs).
  
- `lr=1.0f-2`:              Learning rate. Must be &gt; 0. A lower `eta` results in slower learning, typically requiring a higher `nrounds`.   
  
- `wd=0.f0`:                Weight decay applied to the gradients by the optimizer.
  
- `batchsize=2048`:         Batch size.
  
- `rng=123`:                Either an integer used as a seed to the random number generator or an actual random number generator (`::Random.AbstractRNG`).
  
- `device=:cpu`:            Device on which to perform the computation, either `:cpu` or `:gpu`
  
- `gpuID=0`:                GPU device to use, only relveant if `device = :gpu` 
  

**Internal API**

Do `config = NeuroTabClassifier()` to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTabClassifier(depth=5, ...)`.

**Training model**

A model is trained using [`fit`](/API#MLJModelInterface.fit):

```julia
m = fit(config, dtrain; feature_names, target_name, kwargs...)
```


**Inference**

Models act as a functor. returning predictions when called as a function with features as argument:

```julia
m(data)
```


**MLJ Interface**

From MLJ, the type can be imported using:

```julia
NeuroTabClassifier = @load NeuroTabClassifier pkg=NeuroTabModels
```


Do `model = NeuroTabClassifier()` to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTabClassifier(loss=...)`.

**Training model**

In MLJ or MLJBase, bind an instance `model` to data with     `mach = machine(model, X, y)` where
- `X`: any table of input features (eg, a `DataFrame`) whose columns each have one of the following element scitypes: `Continuous`, `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
  
- `y`: is the target, which can be any `AbstractVector` whose element scitype is `<:Finite`; check the scitype with `scitype(y)`
  

Train the machine using `fit!(mach, rows=...)`.

**Operations**
- `predict(mach, Xnew)`: return predictions of the target given features `Xnew` having the same scitype as `X` above.
  

**Fitted parameters**

The fields of `fitted_params(mach)` are:
- `:fitresult`: The `NeuroTabModel` object.
  

**Report**

The fields of `report(mach)` are:
- `:features`: The names of the features encountered in training.
  

**Examples**

**Internal API**

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


**MLJ Interface**

```julia
using MLJBase, NeuroTabModels
m = NeuroTabClassifier(depth=5, nrounds=10)
X, y = @load_crabs
mach = machine(m, X, y) |> fit!
p = predict(mach, X)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/Evovest/NeuroTabModels.jl/blob/cc9c3677afe6b703742d9667d6cfda07ba24784b/src/learners.jl#L224-L330" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## NeuroTabModel {#NeuroTabModel}
<details class='jldocstring custom-block' open>
<summary><a id='NeuroTabModels.Models.NeuroTabModel' href='#NeuroTabModels.Models.NeuroTabModel'><span class="jlbinding">NeuroTabModels.Models.NeuroTabModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
NeuroTabModel
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/Evovest/NeuroTabModels.jl/blob/cc9c3677afe6b703742d9667d6cfda07ba24784b/src/models/models.jl#L11-L13" target="_blank" rel="noreferrer">source</a></Badge>

</details>

