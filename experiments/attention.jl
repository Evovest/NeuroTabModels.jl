using NeuroTabModels
using DataFrames
using BenchmarkTools
using Dates
using Random: seed!
import Base: length, getindex
import MLUtils: DataLoader
using Lux

Threads.nthreads()

seed!(123)
nobs = Int(1e3)
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

m = NeuroTabModels.fit(
    learner,
    dtrain;
    # deval=dtrain, # FIXME: very slow when deval is used / crashed on GPU
    target_name,
    feature_names,
    weight_name,
    print_every_n=2,
);

# Reactant CPU: 0.952495 seconds (57.96 k allocations: 1.517 GiB, 0.23% gc time, 0.00% compilation time)
# Reactant CPU: 10.326071 seconds (29.30 k allocations: 13.145 GiB, 1.97% gc time)
# FIXME: need to adapt infer: returns only full batches: length of p_train must be == nrow(dtrain)
@time p_train = m(dtrain; device=:gpu);

dfg = groupby(dtrain, :date)
data = NeuroTabModels.Data.get_df_loader_train(dfg; feature_names, target_name, weight_name)
length(data)

n = length(dfg)
nfeats = length(feature_names)
bs = maximum(dfg.ends .- dfg.starts) + 1
x = [zeros(Float32, nfeats, bs) for _ in 1:n]
@time x = [x[i][:, 1:nrow(df)] .= Matrix(df[!, feature_names])' for (i, df) in enumerate(dfg)]
