using NeuroTabModels
using DataFrames
using BenchmarkTools
using Dates
using Random: seed!
import Base: length, getindex
import MLUtils: DataLoader
using Lux
using Statistics

Threads.nthreads()

seed!(123)
nobs = Int(1e4)
num_feat = Int(10)
X = rand(Float32, nobs, num_feat)
Y = randn(Float32, size(X, 1))
dtrain = DataFrame(X, :auto)
feature_names = names(dtrain)
dtrain.y = Y
target_name = "y"
dtrain.w = rand(nrow(dtrain))
weight_name = "w"
dtrain.date = rand(Date("2026-01-01"):Date("2026-01-05"), nrow(dtrain))

arch = NeuroTabModels.NeuroTreeConfig(;
    tree_type=:binary,
    actA=:identity,
    init_scale=1.0,
    depth=4,
    ntrees=16,
    stack_size=1,
    hidden_size=1,
    scaler=true,
)

learner = NeuroTabRegressor(
    arch;
    loss=:mse,
    nrounds=10,
    lr=1e-2,
    batchsize=2048,
    device=:gpu
)

sort!(dtrain, :date)
m = NeuroTabModels.fit(
    learner,
    dtrain;
    deval=dtrain,
    target_name,
    feature_names,
    weight_name,
    print_every_n=2,
);
p_train_1 = m(dtrain; device=:gpu);
p_train_2 = m(dtrain; device=:gpu);

m = NeuroTabModels.fit(
    learner,
    dtrain;
    deval=dtrain,
    target_name,
    feature_names,
    weight_name,
    group_key=:date,
    print_every_n=2,
);
p_train_grp = m(dtrain; device=:gpu);

cor(p_train_1, p_train_2)
cor(p_train_1, p_train_grp)

dfg = groupby(dtrain, :date)
data = NeuroTabModels.Data.get_df_loader_train(dfg; feature_names, target_name, weight_name)
length(data)
