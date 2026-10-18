

# Benchmark: iris

Iris (OpenML 61): species of 150 flowers, 3 classes.

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

    (task = :multiclass, train = 105, eval = 22, test = 23, features = 4)

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

| **model** | **mlogloss** | **accuracy** | **best_iter** | **train_time** | **time_per_round** | **predict_time** |
|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.133708 | 0.956522 | 11.0 | 1.82792 | 0.0589653 | 0.182907 |
| ResNet | 0.172913 | 0.956522 | 10.0 | 3.33591 | 0.111197 | 0.209479 |
| NeuroTree | 0.0267331 | 1.0 | 285.0 | 1.72709 | 0.00566259 | 0.237566 |
| TabM | 0.0323744 | 0.956522 | 152.0 | 2.44334 | 0.0142054 | 0.172878 |

Run on NeuroTabModels 0.5.0 (d547a0f), Julia 1.12.6, gpu, reactant,
2026-10-18 17:09.

## Learning curves

The eval metric at each round. Training stops after 20 rounds without
improvement.

![](bench-iris_files/figure-commonmark/cell-9-output-1.png)
