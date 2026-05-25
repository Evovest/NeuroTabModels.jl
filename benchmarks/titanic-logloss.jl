using MLDatasets
using DataFrames
using Statistics: mean
using StatsBase: median
using Random
using CategoricalArrays
using OrderedCollections
using NeuroTabModels

using CUDA, cuDNN
using Reactant
using Zygote
using Enzyme

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

# arch = NeuroTabModels.NeuroTreeConfig(;
#     tree_type=:binary,
#     k=1,
#     depth=4,
#     ntrees=16,
#     stack_size=1,
#     hidden_size=1,
#     actA=:identity,
#     init_scale=1.0,
#     scaler=true,
# )

# arch = NeuroTabModels.MOETreeConfig(;
#     tree_type=:binary,
#     k=1,
#     depth=3,
#     ntrees=8,
#     stack_size=1,
#     hidden_size=8,
#     init_scale=1.0,
# )
# arch = NeuroTabModels.TabMConfig(;
#     arch_type=:tabm,
#     k=8,
#     d_block=32,
#     n_blocks=3,
#     dropout=0.1,
#     scaling_init=:normal,
# )

arch = NeuroTabModels.ModernNCAConfig(; d_embedding=16, n_blocks=1, d_block=32, dropout=0.1f0, sample_rate=0.5)

# embedding_config = Dict(
#     :embedding_type => :piecewise,
#     :d_embedding => 8,
#     :activation => nothing,
#     :bins => 16,
#     :frequencies => 16,
# )
embedding_config = NeuroTabModels.EmbeddingLayer(num=NeuroTabModels.BatchNormEmbeddings())

learner = NeuroTabRegressor(
    arch;
    embedding_config,
    loss=:logloss,
    nrounds=200,
    early_stopping_rounds=10,
    lr=3e-3,
    device=:cpu,
    backend=:reactant
)

@time m = NeuroTabModels.fit(
    learner,
    dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n=10,
);

p_train = m(dtrain)
p_eval = m(deval)
@info mean((p_train .> 0.5) .== (dtrain[!, target_name] .> 0.5))
@info mean((p_eval .> 0.5) .== (deval[!, target_name] .> 0.5))
