using NeuroTabModels
using DataFrames
using BenchmarkTools
using Random: seed!

Threads.nthreads()

seed!(123)
nobs = Int(1e6)
num_feat = Int(100)
@info "testing with: $nobs observations | $num_feat features."
X = rand(Float32, nobs, num_feat)
Y = randn(Float32, size(X, 1))
dtrain = DataFrame(X, :auto)
feature_names = names(dtrain)
dtrain.y = Y
target_name = "y"

arch = NeuroTabModels.NeuroTreeConfig(;
    tree_type=:binary,
    proj_size=1,
    actA=:identity,
    init_scale=1.0,
    depth=4,
    ntrees=32,
    stack_size=1,
    hidden_size=1,
    scaler=false,
)
# arch = NeuroTabModels.MLPConfig(;
#     act=:relu,
#     stack_size=1,
#     hidden_size=64,
# )

learner = NeuroTabRegressor(
    arch;
    loss=:mse,
    nrounds=10,
    lr=1e-2,
    batchsize=2048,
    device=:gpu
)

# desktop gpu - no-eval: 11.239302 seconds (30.63 M allocations: 6.101 GiB, 3.60% gc time)
# desktop gpu - eval: 15.708276 seconds (23.57 k allocations: 13.156 GiB, 2.02% gc time)
#  13.557744 seconds (26.40 M allocations: 5.989 GiB, 9.60% gc time)
@time m = NeuroTabModels.fit(
    learner,
    dtrain;
    # deval=dtrain,
    target_name,
    feature_names,
    print_every_n=10,
);

# desktop gpu: 0.947362 seconds (484.31 k allocations: 1.526 GiB, 19.83% gc time)
# desktop cpu: 15.708276 seconds (23.57 k allocations: 13.156 GiB, 2.02% gc time)
@time p_train = m(dtrain; device=:cpu);
