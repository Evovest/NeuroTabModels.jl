
# API {#API}

## Training {#Training}
<details class='jldocstring custom-block' open>
<summary><a id='MLJModelInterface.fit' href='#MLJModelInterface.fit'><span class="jlbinding">MLJModelInterface.fit</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
function fit(
    config::NeuroTypes,
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
    device=:cpu,
    gpuID=0,
)
```


Training function of NeuroTabModels&#39; internal API.

**Arguments**
- `config::LearnerTypes`
  
- `dtrain`: Must be `<:AbstractDataFrame`  
  

**Keyword arguments**
- `feature_names`:          Required kwarg, a `Vector{Symbol}` or `Vector{String}` of the feature names.
  
- `target_name`             Required kwarg, a `Symbol` or `String` indicating the name of the target variable.  
  
- `weight_name=nothing`
  
- `offset_name=nothing`
  
- `deval=nothing`           Data for tracking evaluation metric and perform early stopping.
  
- `print_every_n=9999`
  
- `verbosity=1`
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/Evovest/NeuroTabModels.jl/blob/cc9c3677afe6b703742d9667d6cfda07ba24784b/src/Fit/fit.jl#L72-L105" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Inference {#Inference}
<details class='jldocstring custom-block' open>
<summary><a id='NeuroTabModels.Infer.infer' href='#NeuroTabModels.Infer.infer'><span class="jlbinding">NeuroTabModels.Infer.infer</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



infer(m::NeuroTabModel, data)

Return the inference of a `NeuroTabModel` over `data`, where `data` is `AbstractDataFrame`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/Evovest/NeuroTabModels.jl/blob/cc9c3677afe6b703742d9667d6cfda07ba24784b/src/infer.jl#L21-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

