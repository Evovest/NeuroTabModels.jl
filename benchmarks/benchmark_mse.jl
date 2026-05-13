using NeuroTabModels
using DataFrames
using BenchmarkTools
using Random: seed!

using CUDA, cuDNN
using Reactant
using Zygote
using Enzyme

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
    actA=:identity,
    init_scale=1.0,
    depth=4,
    ntrees=32,
    stack_size=1,
    hidden_size=1,
    scaler=false,
)
# arch = NeuroTabModels.TabMConfig(;
#     arch_type=:tabm,
#     k=16,
#     d_block=64,
#     n_blocks=3,
#     dropout=0.1,
#     bins=nothing,
#     use_embeddings=false,
#     embedding_type=:periodic,
#     d_embedding=16,
#     scaling_init=:random_signs,
# )
# arch = NeuroTabModels.MLPConfig(;
#     act=:relu,
#     stack_size=1,
#     hidden_size=64,
# )

# embedding_config = Dict(
#     :embedding_type => :linear,
#     :d_embedding => 8,
#     :activation => "identity",
# )
embedding_config = NeuroTabModels.EmbeddingLayer(num=NeuroTabModels.BatchNormEmbeddings())

learner = NeuroTabRegressor(
    arch;
    embedding_config,
    loss=:mse,
    nrounds=10,
    lr=1e-2,
    batchsize=2048,
    device=:gpu,
    backend=:enzyme
)

# Reactant GPU: 5.970480 seconds (2.33 M allocations: 5.242 GiB, 3.80% gc time, 0.00% compilation time)
# Reactant GPU with eval: 10.154589 seconds (2.33 M allocations: 10.563 GiB, 17.66% gc time, 0.00% compilation time: 100% of which was recompilation)
# Zygote GPU: 8.798624 seconds (15.09 M allocations: 5.728 GiB, 13.60% gc time)
# Zygote GPU with eval: 13.715713 seconds (20.61 M allocations: 11.236 GiB, 22.54% gc time)
# Zygote CPU: 338.912009 seconds (62.93 M allocations: 373.382 GiB, 10.93% gc time, 6.38% compilation time: <1% of which was recompilation)
# Enzyme CPU: 657.713208 seconds (1.98 M allocations: 270.987 GiB, 6.37% gc time)
@time m = NeuroTabModels.fit(
    learner,
    dtrain;
    # deval=dtrain, # FIXME: very slow when deval is used / crashed on GPU
    target_name,
    feature_names,
    print_every_n=2,
);

# Reactant CPU: 0.952495 seconds (57.96 k allocations: 1.517 GiB, 0.23% gc time, 0.00% compilation time)
# Reactant CPU: 10.326071 seconds (29.30 k allocations: 13.145 GiB, 1.97% gc time)
# FIXME: need to adapt infer: returns only full batches: length of p_train must be == nrow(dtrain)
@time p_train = m(dtrain; device=:gpu);
