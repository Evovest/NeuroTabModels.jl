using Random
using CSV
using DataFrames
using Statistics: mean, std
using StatsBase: tiedrank

using Enzyme
using Reactant
# using CUDA, cuDNN
# using Zygote

using NeuroTabModels
using AWS: AWSCredentials, AWSConfig, @service
@service S3

nobs = 1_000_000
nfeats = 100
df_tot = DataFrame(rand(Float32, nobs, nfeats), :auto)
feature_names = names(df_tot)

df_tot.y .= randn(nobs)
target_name = "y"

df_tot.grp = rand(1:round(Int, nrow(df_tot) / 2500), nrow(df_tot))

train_idx = 1:floor(Int, nobs * 0.8)
eval_idx = setdiff(1:nobs, train_idx)
dtrain = df_tot[train_idx, :];
deval = df_tot[eval_idx, :];

sort!(dtrain, :grp)
sort!(deval, :grp)

arch = NeuroTabModels.NeuroTreeConfig(;
    tree_type=:binary,
    actA=:identity,
    k=1,
    ntrees=32,
    depth=4,
    stack_size=1,
    hidden_size=16,
    init_scale=0.1,
    scaler=true,
)

# arch = NeuroTabModels.MLPConfig(;
#     act=:relu,
#     stack_size=1,
#     hidden_size=256,
# )

device = :gpu
backend = :reactant
loss = :mse # :mse :gaussian_mle :tweedie
metric = :correlation # :mse :gaussian_mle :tweedie

# embedding_config = Dict(
#     :embedding_type => :piecewise,
#     :d_embedding => 8,
#     :activation => nothing,
#     :bins => 16,
#     :frequencies => 16,
# )
embedding_config = Dict(:embedding_type => :linear, :d_embedding => 1, :activation => "identity")
# embedding_config = Dict(:embedding_type => "batchnorm")

learner = NeuroTabRegressor(
    arch; embedding_config, loss, metric, nrounds=100, early_stopping_rounds=100, lr=1e-4, batchsize=0, device, backend
)

group_name = "grp" #"grp" # nothing
# group_name = nothing #"grp" # nothing
@time m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names, group_name, print_every_n=5);
# @time m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names, weight_name, group_name, print_every_n=5);
# @time m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, group_name, print_every_n=5);

p_eval = m(deval; device=:cpu);
p_eval = p_eval[:, 1]
mse_eval = mean((p_eval .- deval.y) .^ 2)
@info "MSE - deval" mse_eval
