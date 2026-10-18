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
`julia --project=quarto quarto/benchmarks/run.jl`. Add a dataset as a
new entry in `DATASETS` in `quarto/benchmarks/benchmarks.jl`.

## boston

[Report](bench-boston.md). Boston housing (OpenML 531): median home
value of 506 Boston census tracts.

Version 0.5.0 on gpu with reactant.

| **model** | **rmse** | **mae** | **best_iter** | **train_time** | **time_per_round** | **predict_time** |
|---:|---:|---:|---:|---:|---:|---:|
| MLP | 4.567 | 2.641 | 47 | 1.824 | 0.02722 | 0.1143 |
| ResNet | 4.408 | 2.494 | 31 | 3.218 | 0.0631 | 0.1365 |
| NeuroTree | 4.267 | 2.512 | 146 | 1.834 | 0.01105 | 0.2401 |
| TabM | 3.875 | 2.4 | 44 | 2.676 | 0.04181 | 0.1501 |

## california

[Report](bench-california.md). California housing (OpenML 44977): median
house value of 20,640 block groups.

Version 0.5.0 on gpu with reactant.

| **model** | **rmse** | **mae** | **best_iter** | **train_time** | **time_per_round** | **predict_time** |
|---:|---:|---:|---:|---:|---:|---:|
| MLP | 52610.0 | 37150.0 | 84 | 2.398 | 0.02306 | 0.212 |
| ResNet | 52780.0 | 35630.0 | 25 | 3.69 | 0.082 | 0.2694 |
| NeuroTree | 51530.0 | 34610.0 | 173 | 2.003 | 0.01038 | 0.6252 |
| TabM | 49110.0 | 33560.0 | 159 | 4.008 | 0.02239 | 0.2808 |

## higgs

[Report](bench-higgs.md). Higgs (OpenML 23512): signal vs background for
98k collision events.

Version 0.5.0 on gpu with reactant.

| **model** | **logloss** | **accuracy** | **best_iter** | **train_time** | **time_per_round** | **predict_time** |
|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.5549 | 0.7138 | 11 | 2.696 | 0.08696 | 0.2442 |
| ResNet | 0.5947 | 0.7074 | 6 | 4.855 | 0.1867 | 0.294 |
| NeuroTree | 0.5317 | 0.7291 | 31 | 2.464 | 0.04832 | 0.6919 |
| TabM | 0.5133 | 0.7452 | 96 | 8.184 | 0.08184 | 0.3399 |

## iris

[Report](bench-iris.md). Iris (OpenML 61): species of 150 flowers, 3
classes.

Version 0.5.0 on gpu with reactant.

| **model** | **mlogloss** | **accuracy** | **best_iter** | **train_time** | **time_per_round** | **predict_time** |
|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.1337 | 0.9565 | 11 | 1.862 | 0.06005 | 0.1829 |
| ResNet | 0.1729 | 0.9565 | 10 | 3.387 | 0.1129 | 0.2071 |
| NeuroTree | 0.02601 | 1 | 359 | 1.941 | 0.005121 | 0.2242 |
| TabM | 0.03237 | 0.9565 | 152 | 2.48 | 0.01442 | 0.1726 |

## titanic

[Report](bench-titanic.md). Titanic (OpenML 40945): survival of 1,309
passengers.

Version 0.5.0 on gpu with reactant.

| **model** | **logloss** | **accuracy** | **best_iter** | **train_time** | **time_per_round** | **predict_time** |
|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.3685 | 0.8477 | 4 | 1.817 | 0.0757 | 0.1108 |
| ResNet | 0.3925 | 0.8426 | 6 | 3.299 | 0.1269 | 0.1429 |
| NeuroTree | 0.365 | 0.8579 | 69 | 2.035 | 0.02287 | 0.2495 |
| TabM | 0.4115 | 0.8223 | 16 | 2.071 | 0.05752 | 0.1459 |
