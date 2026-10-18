# Benchmarks


Each dataset has a report that fits MLP, ResNet, NeuroTree and TabM with
fixed settings and scores them on a held-out test set. Every report run
adds its scores to `quarto/benchmarks/results.csv`, tagged with the
NeuroTabModels version and git commit.

This page shows, for each dataset, the test scores and timings of the
latest version, with the change from the previous version in brackets. A
version run more than once counts with its latest run. Timings leave out
compilation but depend on the machine, so runs are meant to be made on
the same one.

To add a run, render the reports with
`julia --project=quarto quarto/benchmarks/run.jl` followed by the
datasets. Add a dataset as a new entry in `DATASETS` in
`quarto/benchmarks/benchmarks.jl`.

## boston

[Report](bench-boston.md). Boston housing (OpenML 531): median home
value of 506 Boston census tracts.

Version 0.5.0 on gpu with reactant.

| **model** | **rmse** | **mae** | **best_iter** | **train_time** | **time_per_round** | **predict_time** |
|---:|---:|---:|---:|---:|---:|---:|
| MLP | 4.567 | 2.641 | 47 | 1.806 | 0.02695 | 0.1081 |
| ResNet | 4.408 | 2.494 | 31 | 3.235 | 0.06344 | 0.136 |
| NeuroTree | 4.025 | 2.355 | 201 | 2.106 | 0.009528 | 0.3405 |
| TabM | 3.875 | 2.4 | 44 | 2.025 | 0.03163 | 0.1415 |

## california

[Report](bench-california.md). California housing (OpenML 44977): median
house value of 20,640 block groups.

Version 0.5.0 on gpu with reactant.

| **model** | **rmse** | **mae** | **best_iter** | **train_time** | **time_per_round** | **predict_time** |
|---:|---:|---:|---:|---:|---:|---:|
| MLP | 52610.0 | 37150.0 | 84 | 2.387 | 0.02295 | 0.2068 |
| ResNet | 52780.0 | 35630.0 | 25 | 3.666 | 0.08147 | 0.2733 |
| NeuroTree | 51740.0 | 35290.0 | 193 | 5.428 | 0.02714 | 0.7159 |
| TabM | 49110.0 | 33560.0 | 159 | 4.875 | 0.02724 | 0.2848 |

## higgs

[Report](bench-higgs.md). Higgs (OpenML 23512): signal vs background for
98k collision events.

Version 0.5.0 on gpu with reactant.

| **model** | **logloss** | **accuracy** | **best_iter** | **train_time** | **time_per_round** | **predict_time** |
|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.5549 | 0.7138 | 11 | 2.663 | 0.08591 | 0.214 |
| ResNet | 0.5947 | 0.7074 | 6 | 4.698 | 0.1807 | 0.2911 |
| NeuroTree | 0.5256 | 0.7363 | 45 | 8.233 | 0.1267 | 1.031 |
| TabM | 0.5133 | 0.7452 | 96 | 7.129 | 0.07129 | 0.3335 |

## iris

[Report](bench-iris.md). Iris (OpenML 61): species of 150 flowers, 3
classes.

Version 0.5.0 on gpu with reactant.

| **model** | **mlogloss** | **accuracy** | **best_iter** | **train_time** | **time_per_round** | **predict_time** |
|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.1337 | 0.9565 | 11 | 1.828 | 0.05897 | 0.1829 |
| ResNet | 0.1729 | 0.9565 | 10 | 3.336 | 0.1112 | 0.2095 |
| NeuroTree | 0.02673 | 1 | 285 | 1.727 | 0.005663 | 0.2376 |
| TabM | 0.03237 | 0.9565 | 152 | 2.443 | 0.01421 | 0.1729 |

## titanic

[Report](bench-titanic.md). Titanic (OpenML 40945): survival of 1,309
passengers.

Version 0.5.0 on gpu with reactant.

| **model** | **logloss** | **accuracy** | **best_iter** | **train_time** | **time_per_round** | **predict_time** |
|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.3685 | 0.8477 | 4 | 1.821 | 0.07589 | 0.1131 |
| ResNet | 0.3925 | 0.8426 | 6 | 3.384 | 0.1302 | 0.1404 |
| NeuroTree | 0.3709 | 0.8528 | 62 | 2.1 | 0.0256 | 0.345 |
| TabM | 0.4115 | 0.8223 | 16 | 2.07 | 0.05749 | 0.1431 |
