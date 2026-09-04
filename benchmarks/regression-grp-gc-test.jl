using Random
using DataFrames
using Statistics: mean

using Enzyme
using Reactant

using NeuroTabModels

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

device = :gpu
backend = :reactant
loss = :mse
metric = :pearson

embedding_config = Dict(:embedding_type => :linear, :d_embedding => 1, :activation => "identity")

learner = NeuroTabRegressor(
    arch; embedding_config, loss, metric, nrounds=200, early_stopping_rounds=20, lr=1e-4, batchsize=0, device, backend
)

group_name = "grp"
@time m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names, group_name, print_every_n=5);

p_eval = m(deval; device=:cpu);
p_eval = p_eval[:, 1]
mse_eval = mean((p_eval .- deval.y) .^ 2)
@info "MSE - deval" mse_eval
@info "SUCCESS: completed all rounds without OOM"
