using MLDatasets
using DataFrames
using Statistics: mean
using StatsBase: median
using CategoricalArrays
using Random
using CategoricalArrays
using OrderedCollections
using NeuroTabModels

Random.seed!(123)

df = MLDatasets.Titanic().dataframe

# convert string feature to Categorical
transform!(df, :Sex => categorical => :Sex)
transform!(df, :Sex => ByRow(levelcode) => :Sex)

# treat string feature and missing values
transform!(df, :Age => ByRow(ismissing) => :Age_ismissing)
transform!(df, :Age => (x -> coalesce.(x, median(skipmissing(x)))) => :Age);

# remove unneeded variables
df = df[:, Not([:PassengerId, :Name, :Embarked, :Cabin, :Ticket])]

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

target_name = "Survived"
feature_names = setdiff(names(df), ["Survived"])

arch = NeuroTabModels.NeuroTreeConfig(;
    tree_type=:binary,
    proj_size=1,
    init_scale=1.0,
    depth=4,
    ntrees=16,
    stack_size=1,
    hidden_size=1,
    actA=:identity,
    MLE_tree_split=false,
)

learner = NeuroTabRegressor(
    arch;
    loss=:logloss, # FIXME: gaussian_mle don't train
    nrounds=100,
    early_stopping_rounds=2,
    lr=3e-2,
    device=:cpu
)

@time m = NeuroTabModels.fit(
    learner,
    dtrain;
    # deval, # FIXME: important slowdown when deval is used
    target_name,
    feature_names,
    print_every_n=10,
);

# commented out - inference not yet adapted
p_train = m(dtrain)
p_eval = m(deval)
@info mean((p_train .> 0.5) .== (dtrain[!, target_name] .> 0.5))
@info mean((p_eval .> 0.5) .== (deval[!, target_name] .> 0.5))
