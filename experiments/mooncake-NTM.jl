using MLDatasets
using DataFrames
using Statistics: mean
using StatsBase: median
using CategoricalArrays
using Random
using CategoricalArrays
using OrderedCollections
using NeuroTabModels

using Mooncake
import Mooncake as MC
import DifferentiationInterface as DI

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
)

learner = NeuroTabRegressor(
    arch;
    loss=:logloss,
    nrounds=200,
    early_stopping_rounds=2,
    lr=3e-2,
    device=:cpu
)



dtrain = NeuroTabModels.Data.get_df_loader_train(df; feature_names, target_name, batchsize=200, device="cpu")
info = Dict(
    :nrounds => 0,
    :feature_names => feature_names
)
chain = learner.arch(; nfeats=length(feature_names), outsize=1)
loss = NeuroTabModels.Losses.get_loss_fn(learner.loss)
L = NeuroTabModels.Losses.get_loss_type(learner.loss)
m = NeuroTabModel(L, chain, info)
# m = m |> gpu

optim = NeuroTabModels.Fit.OptimiserChain(NeuroTabModels.Fit.NAdam(learner.lr))
opts = NeuroTabModels.Fit.Optimisers.setup(optim, m)

x = rand(Float32, 7, 32)
m(x)
y = rand(32)
loss(m, x, y)

############
# Mooncake
############
cache = MC.prepare_gradient_cache(loss, m, x, y);
val, grads = MC.value_and_gradient!!(cache, loss, m, x, y)
NeuroTabModels.Fit.Optimisers.update!(opts, m.chain, grads[2].fields.chain.fields)

using Flux, Optimisers
m = Chain(BatchNorm(3), Dense(3, 1))
myloss(m, x, y) = mean((m(x) .- y) .^ 2)
x = randn(Float32, 3, 4)
y = randn(Float32, 4)
myloss(m, x, y)

############
# Mooncake DI
############
using Mooncake
backend = DI.AutoMooncake(; config=nothing)
typical_x, typical_y = similar(x), similar(y)
prep = DI.prepare_gradient(myloss, backend, m, DI.Constant(typical_x), DI.Constant(typical_y))
val, grads = DI.value_and_gradient(myloss, prep, backend, m, DI.Constant(x), DI.Constant(y))
opts = Optimisers.setup(optim, m)
NeuroTabModels.Fit.Optimisers.update!(opts, m, grads)

############
# Zygote
############
using Zygote
backend = DI.AutoZygote()
typical_x, typical_y = similar(x), similar(y)
prep = DI.prepare_gradient(loss, backend, m, DI.Constant(typical_x), DI.Constant(typical_y))
val, grads = DI.value_and_gradient(loss, prep, backend, m, DI.Constant(x), DI.Constant(y))
opts = NeuroTabModels.Fit.Optimisers.setup(optim, m)
NeuroTabModels.Fit.Optimisers.update!(opts, m, grads)


############
# Enzyme
############
using Enzyme
backend = DI.AutoEnzyme()
typical_x, typical_y = similar(x), similar(y)
prep = DI.prepare_gradient(loss, backend, m, DI.Constant(typical_x), DI.Constant(typical_y))
val, grads = DI.value_and_gradient(loss, prep, backend, m, DI.Constant(x), DI.Constant(y))
opts = NeuroTabModels.Fit.Optimisers.setup(optim, m)
NeuroTabModels.Fit.Optimisers.update!(opts, m, grads)
