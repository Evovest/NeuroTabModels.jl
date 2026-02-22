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

# Reactant GPU: 5.970480 seconds (2.33 M allocations: 5.242 GiB, 3.80% gc time, 0.00% compilation time)
# Zygote GPU: 9.855853 seconds (27.92 M allocations: 6.005 GiB, 3.58% gc time)
#  13.557744 seconds (26.40 M allocations: 5.989 GiB, 9.60% gc time)
@time m = NeuroTabModels.fit(
    learner,
    dtrain;
    deval=dtrain, # FIXME: very slow when deval is used, need to adapt infer
    target_name,
    feature_names,
    print_every_n=2,
);

# desktop: 0.771839 seconds (369.20 k allocations: 1.522 GiB, 5.94% gc time)
# FIXME: need to adapt infer
# @time p_train = m(dtrain; device=:gpu);
