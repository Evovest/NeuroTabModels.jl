using Random
using CSV
using DataFrames
using Statistics: mean, std
using StatsBase: tiedrank
using NeuroTabModels
using AWS: AWSCredentials, AWSConfig, @service
@service S3

aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds=aws_creds, region="ca-central-1")

path = "share/data/year/year.csv"
raw = S3.get_object("jeremiedb", path, Dict("response-content-type" => "application/octet-stream"); aws_config)
df = DataFrame(CSV.File(raw, header=false))
df_tot = copy(df)

path = "share/data/year/year-train-idx.txt"
raw = S3.get_object("jeremiedb", path, Dict("response-content-type" => "application/octet-stream"); aws_config)
train_idx = DataFrame(CSV.File(raw, header=false))[:, 1] .+ 1

path = "share/data/year/year-eval-idx.txt"
raw = S3.get_object("jeremiedb", path, Dict("response-content-type" => "application/octet-stream"); aws_config)
eval_idx = DataFrame(CSV.File(raw, header=false))[:, 1] .+ 1

target_name = "y"
rename!(df_tot, "Column1" => target_name)
feature_names = setdiff(names(df_tot), ["y", "w"])
df_tot.w .= 1.0

# function percent_rank(x::AbstractVector{T}) where {T}
#     return tiedrank(x) / (length(x) + 1)
# end
# transform!(df_tot, feature_names .=> percent_rank .=> feature_names)

dtrain = df_tot[train_idx, :];
deval = df_tot[eval_idx, :];
dtest = df_tot[(end-51630+1):end, :];

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

# arch = NeuroTabModels.MOETreeConfig(;
#     tree_type=:binary,
#     depth=5,
#     ntrees=8,
#     stack_size=1,
#     init_scale=0.1,
# )

# arch = NeuroTabModels.TabMConfig(;
#     arch_type=:tabm,
#     k=16,
#     d_block=64,
#     n_blocks=3,
#     dropout=0.1,
#     # scaling_init=:normal,
# )
# arch = NeuroTabModels.MLPConfig(;
#     act=:relu,
#     stack_size=1,
#     hidden_size=256,
# )
# arch = NeuroTabModels.ResNetConfig(;
#     num_blocks=1,
#     hidden_size=128,
#     act=:relu,
#     dropout=0.5,
#     MLE_tree_split=false
# )

device = :gpu
loss = :mse # :mse :gaussian_mle :tweedie

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
    loss,
    nrounds=200,
    early_stopping_rounds=2,
    lr=1e-3,
    batchsize=512,
    device
)

@time m = NeuroTabModels.fit(
    learner,
    dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n=5,
);

p_eval = m(deval; device=:cpu);
p_eval = p_eval[:, 1]
mse_eval = mean((p_eval .- deval.y) .^ 2)
@info "MSE - deval" mse_eval

p_test = m(dtest; device=:cpu);
p_test = p_test[:, 1]
mse_test = mean((p_test .- dtest.y) .^ 2)
@info "MSE - dtest" mse_test
