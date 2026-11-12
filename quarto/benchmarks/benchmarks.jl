# Datasets, metrics and the results file shared by `bench-template.qmd` and `bench-summary.qmd`.

using NeuroTabModels
using DataFrames
using CategoricalArrays
using CSV
using Dates
using OpenML
using PrettyTables
using Random
using Statistics: mean, std, median

const RESULTS_FILE = joinpath(@__DIR__, "results.csv")

#===========================================================================================
Datasets: each loader returns a DataFrame of numeric features plus the target column, and the
offset column for a dataset with an `offset`.
Add a dataset by adding an entry to `DATASETS` and its name to `datasets` in `run.jl`.
===========================================================================================#

function _boston()
    df = DataFrame(OpenML.load(531))
    df.CHAS, df.RAD = parse.(Float64, String.(df.CHAS)), parse.(Float64, String.(df.RAD))
    return df
end

function _titanic()
    df = DataFrame(OpenML.load(40945))
    df.sex = Float64.(df.sex .== "male")
    df.age_ismissing = Float64.(ismissing.(df.age))
    df.age = coalesce.(df.age, median(skipmissing(df.age)))
    df.fare = coalesce.(df.fare, median(skipmissing(df.fare)))
    df.survived = parse.(Float64, String.(df.survived))
    return select(df, Not([:name, :embarked, :ticket, :cabin, :boat, :body, Symbol("home.dest")]))
end

function _higgs()
    df = dropmissing(DataFrame(OpenML.load(23512)))
    df.class = parse.(Float64, String.(df.class))
    return df
end

# The numeric policy features, with extreme values capped. `Exposure` is not a feature but the
# offset: a policy active for a full year has more chances to claim than one active for a month.
function _fremtpl()
    df = DataFrame(OpenML.load(41214))
    return DataFrame(;
        df.DrivAge, df.VehPower,
        VehAge=min.(df.VehAge, 20), BonusMalus=min.(df.BonusMalus, 150), LogDensity=log.(df.Density),
        ClaimNb=min.(df.ClaimNb, 4), LogExposure=log.(min.(df.Exposure, 1)),
    )
end

const DATASETS = Dict(
    "boston" => (;
        description="Boston housing (OpenML 531): median home value of 506 Boston census tracts.",
        task=:regression, target="MEDV", nrounds=400, batchsize=64,
        load=_boston,
    ),
    "titanic" => (;
        description="Titanic (OpenML 40945): survival of 1,309 passengers.",
        task=:binary, target="survived", nrounds=400, batchsize=64,
        load=_titanic,
    ),
    "iris" => (;
        description="Iris (OpenML 61): species of 150 flowers, 3 classes.",
        task=:multiclass, target="class", nrounds=400, batchsize=16,
        load=() -> DataFrame(OpenML.load(61)),
    ),
    "california" => (;
        description="California housing (OpenML 44977): median house value of 20,640 block groups.",
        task=:regression, target="medianHouseValue", nrounds=200, batchsize=1024,
        load=() -> DataFrame(OpenML.load(44977)),
    ),
    "higgs" => (;
        description="Higgs (OpenML 23512): signal vs background for 98k collision events.",
        task=:binary, target="class", nrounds=100, batchsize=1024,
        load=_higgs,
    ),
    "fremtpl" => (;
        description="freMTPL2freq (OpenML 41214): claim counts of 678k French motor insurance policies, with the log exposure as offset.",
        task=:count, target="ClaimNb", offset="LogExposure", nrounds=100, batchsize=2048,
        load=_fremtpl,
    ),
    "year" => (;
        description="YearPredictionMSD (OpenML 44027): release year of 515k songs from 90 audio features.",
        task=:regression, target="year", nrounds=100, batchsize=2048,
        load=() -> DataFrame(OpenML.load(44027)),
    ),
    "nyc_taxi" => (;
        description="NYC green taxi, December 2016 (OpenML 44143): tip amount of 582k trips.",
        task=:regression, target="tipamount", nrounds=100, batchsize=2048,
        load=() -> DataFrame(OpenML.load(44143)),
    ),
    "zurich" => (;
        description="Zurich public transport delays (OpenML 45034): delay of 5.5M stops, from weather and time of day.",
        task=:regression, target="delay", nrounds=50, batchsize=4096,
        load=() -> DataFrame(OpenML.load(45034)),
    ),
)

"""
    load_benchmark(name; seed=123)

Load dataset `name` and split it 70/15/15 into train, eval and test sets. Features are
standardised with the train mean and std.
"""
function load_benchmark(name; seed=123)
    spec = DATASETS[name]
    df = spec.load()
    target_name, offset_name = spec.target, get(spec, :offset, nothing)
    feature_names = setdiff(names(df), [target_name, offset_name])
    for f in feature_names
        df[!, f] = Float64.(df[!, f])
    end

    n = nrow(df)
    perm = randperm(Xoshiro(seed), n)
    n_train, n_eval = round(Int, 0.7n), round(Int, 0.15n)
    dtrain = df[perm[1:n_train], :]
    deval = df[perm[(n_train + 1):(n_train + n_eval)], :]
    dtest = df[perm[(n_train + n_eval + 1):end], :]

    for f in feature_names
        μ, σ = mean(dtrain[!, f]), std(dtrain[!, f])
        σ = σ > 0 ? σ : 1.0
        for d in (dtrain, deval, dtest)
            d[!, f] = (d[!, f] .- μ) ./ σ
        end
    end
    return (; spec, dtrain, deval, dtest, feature_names, target_name, offset_name)
end

const LOSSES = Dict(:regression => :mse, :binary => :logloss, :count => :poisson)

# a regressor with the task's loss, or a classifier for several classes
learner(task, arch; kwargs...) = task == :multiclass ? NeuroTabClassifier(arch; kwargs...) :
    NeuroTabRegressor(arch; loss=LOSSES[task], kwargs...)

#===========================================================================================
Test metrics, on the scale of the target.
===========================================================================================#

test_metrics(task::Symbol, p, y) = test_metrics(Val(task), p, y)
function test_metrics(::Val{:regression}, p, y)
    return ["rmse" => sqrt(mean((p .- y) .^ 2)), "mae" => mean(abs.(p .- y))]
end
function test_metrics(::Val{:binary}, p, y)
    p = clamp.(p, 1e-7, 1 - 1e-7)
    logloss = -mean(y .* log.(p) .+ (1 .- y) .* log.(1 .- p))
    return ["logloss" => logloss, "accuracy" => mean((p .> 0.5) .== (y .> 0.5))]
end
function test_metrics(::Val{:multiclass}, p, y)
    k = levelcode.(y)
    mlogloss = -mean(log(max(p[i, k[i]], 1e-7)) for i in eachindex(k))
    return ["mlogloss" => mlogloss, "accuracy" => mean(argmax.(eachrow(p)) .== k)]
end
# `p` is the expected count; `total_ratio` is the predicted total over the observed total
function test_metrics(::Val{:count}, p, y)
    deviance = 2 * mean(y .* log.(max.(y, 1e-7) ./ p) .- (y .- p))
    return ["deviance" => deviance, "total_ratio" => sum(p) / sum(y)]
end

#===========================================================================================
Results file: one row per (run, dataset, model, metric), appended by each report.
===========================================================================================#

"""
    run_info()

Package version and git commit of the NeuroTabModels being benchmarked. The commit gets a
`-dirty` suffix when `src/` or `Project.toml` have uncommitted changes.
"""
function run_info()
    dir = pkgdir(NeuroTabModels)
    commit = readchomp(`git -C $dir rev-parse --short HEAD`)
    dirty = !isempty(readchomp(`git -C $dir status --porcelain -- src Project.toml`))
    return (;
        date=Dates.format(now(), "yyyy-mm-dd HH:MM"),
        version=string(pkgversion(NeuroTabModels)),
        commit=dirty ? "$commit-dirty" : commit,
        julia=string(VERSION),
    )
end

"""
    record_results(rows)

Write a run's rows to `RESULTS_FILE`, replacing earlier rows for the same commit, device,
backend and dataset.
"""
function record_results(rows::DataFrame)
    old = isfile(RESULTS_FILE) ? read_results() : similar(rows, 0)
    r = first(rows)
    keep = .!((old.commit .== r.commit) .& (old.device .== r.device) .& (old.backend .== r.backend) .& (old.dataset .== r.dataset))
    CSV.write(RESULTS_FILE, vcat(old[keep, :], rows))
end

read_results() = CSV.read(RESULTS_FILE, DataFrame; types=Dict(:commit => String, :version => String))
