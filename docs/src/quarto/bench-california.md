

# Benchmark: california

California housing (OpenML 44977): median house value of 20,640 block
groups.

This report fits each architecture on the same data with the same
settings, then scores it on a held-out test set. Running it adds its
results to `quarto/benchmarks/results.csv`, which feeds the [benchmark
summary](bench-summary.md).

## Data

The data is split 70/15/15 into train, eval and test sets. Features are
standardised with the train mean and std. The eval set drives early
stopping, and the test set is only used for the final scores.

``` julia
data = load_benchmark(dataset)
(; spec, dtrain, deval, dtest, feature_names, target_name) = data
(; task=spec.task, train=nrow(dtrain), eval=nrow(deval), test=nrow(dtest), features=length(feature_names))
```

    (task = :regression, train = 14448, eval = 3096, test = 3096, features = 8)

## Models

Each architecture is close to its defaults, and training uses the
package defaults (learning rate, loss for the task) apart from the
number of rounds, the batch size and early stopping.

``` julia
archs = [
    "MLP" => MLPConfig(; hidden_size=64, stack_size=2),
    "ResNet" => ResNetConfig(; hidden_size=64, stack_size=2),
    "NeuroTree" => NeuroTreeConfig(; k=8, depth=4, ntrees=32),
    "TabM" => TabMConfig(; k=8, d_block=128, n_blocks=2),
]
settings = (; nrounds=spec.nrounds, batchsize=spec.batchsize, early_stopping_rounds=20, device, backend)
```

## Training

Each model is fitted once for one round first, so the timings below
leave out compilation.

``` julia
models, rows = Dict{String,Any}(), DataFrame()
info = run_info()
for (name, arch) in archs
    config = learner(spec.task, arch; settings...)
    warmup = learner(spec.task, arch; settings..., nrounds=1)
    NeuroTabModels.fit(warmup, dtrain; feature_names, target_name, deval, verbosity=0)

    train_time = @elapsed m = NeuroTabModels.fit(config, dtrain; feature_names, target_name, deval, verbosity=0)
    m(dtest)
    predict_time = @elapsed p = m(dtest)
    models[name] = m

    metrics = vcat(
        test_metrics(spec.task, p, dtest[!, target_name]),
        [
            "best_iter" => m.info[:logger][:best_iter],
            "train_time" => train_time,
            # early stopping sets the number of rounds, so speed is compared per round
            "time_per_round" => train_time / max(m.info[:logger][:nrounds], 1),
            "predict_time" => predict_time,
        ],
    )
    for (metric, value) in metrics
        push!(rows, (; info..., device, backend, dataset, model=name, metric, value=Float64(value)))
    end
end
record_results(rows)
```

## Results

Test-set scores, and training (in total and per round) and prediction
times in seconds.

| **model** | **rmse** | **mae** | **best_iter** | **train_time** | **time_per_round** | **predict_time** |
|---:|---:|---:|---:|---:|---:|---:|
| MLP | 52614.7 | 37151.3 | 84.0 | 2.38659 | 0.022948 | 0.206756 |
| ResNet | 52777.5 | 35632.1 | 25.0 | 3.66605 | 0.0814679 | 0.273297 |
| NeuroTree | 51739.2 | 35288.3 | 193.0 | 5.42781 | 0.0271391 | 0.715881 |
| TabM | 49108.5 | 33555.3 | 159.0 | 4.87542 | 0.027237 | 0.284845 |

Run on NeuroTabModels 0.5.0 (d547a0f), Julia 1.12.6, gpu, reactant,
2026-10-18 17:14.

## Learning curves

The eval metric at each round. Training stops after 20 rounds without
improvement.

![](bench-california_files/figure-commonmark/cell-9-output-1.png)
