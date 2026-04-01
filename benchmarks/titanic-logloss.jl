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
    k=8,
    depth=4,
    ntrees=16,
    stack_size=2,
    hidden_size=8,
    actA=:identity,
    init_scale=1.0,
    scaler=true,
)
# arch = NeuroTabModels.TabMConfig(;
#     arch_type=:tabm,
#     k=8,
#     d_block=32,
#     n_blocks=3,
#     dropout=0.1,
#     scaling_init=:normal,
# )

# embedding_config = Dict(
#     :embedding_type => :piecewise,
#     :d_embedding => 8,
#     :activation => nothing,
#     :bins => 16,
#     :frequencies => 16,
# )
embedding_config = Dict(:embedding_type => :batchnorm)

learner = NeuroTabRegressor(
    arch;
    embedding_config,
    loss=:logloss,
    nrounds=100,
    early_stopping_rounds=2,
    lr=1e-2,
    device=:gpu
)

@time m = NeuroTabModels.fit(
    learner,
    dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n=10,
);

# commented out - inference not yet adapted
p_train = m(dtrain)
p_eval = m(deval)
@info mean((p_train .> 0.5) .== (dtrain[!, target_name] .> 0.5))
@info mean((p_eval .> 0.5) .== (deval[!, target_name] .> 0.5))

# using Lux
# using NeuroTabModels.Models.NeuroTrees: NeuroTree, StackedNeuroTree
# rng = Random.default_rng()
# x1 = randn(2, 5)
# m1 = NeuroTree(2 => 1, depth=3, trees=4, k=1, tree_type=:binary)
# m1 = NeuroTree(2 => 1, depth=3, trees=4, k=1, tree_type=:oblivious)
# ps, st = Lux.setup(rng, m1)
# y, st = m1(x1, ps, st)
# size(y)
# size(st.ml)
# size(st.ms)

# m1 = StackedNeuroTree(2 => 3; hidden_size=5, stack_size=2, depth=3, trees=7, k=4)
# ps, st = Lux.setup(rng, m1)
# y, st = m1(x1, ps, st)
# y, st = m1(x1, ps, st)
# size(y)
# size(st.ml)
# size(st.ms)
# m1 = Chain(Dense(2 => 6), Dense(6 => 3), StackedNeuroTree(3 => 3; hidden_size=5, stack_size=2, depth=3, trees=7, k=4))
# ps, st = Lux.setup(rng, m1)
# y, st = m1(x1, ps, st)
# size(y)
# length(m1)
# m2 = m1[1:2]
# ps, st = Lux.setup(rng, m2)
# y, st = m2(x1, ps, st)
# size(y)
