using Random
using CSV
using DataFrames
using Statistics: mean, std
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

transform!(df_tot, "Column1" => identity => "y_raw")
select!(df_tot, Not("Column1"))
transform!(df_tot, "y_raw" => (x -> (x .- mean(x)) ./ std(x)) => "y_norm")
# transform!(df_tot, "y_raw" => (x -> (x .- minimum(x)) ./ std(x)) => "y_norm")
feature_names = setdiff(names(df_tot), ["y_raw", "y_norm", "w"])
df_tot.w .= 1.0
target_name = "y_norm"

# function percent_rank(x::AbstractVector{T}) where {T}
#     return tiedrank(x) / (length(x) + 1)
# end

# transform!(df_tot, feature_names .=> percent_rank .=> feature_names)

dtrain = df_tot[train_idx, :];
deval = df_tot[eval_idx, :];
dtest = df_tot[(end-51630+1):end, :];

arch = NeuroTabModels.NeuroTreeConfig(;
    tree_type=:binary,
    proj_size=4,
    actA=:identity,
    depth=4,
    ntrees=32,
    stack_size=1,
    hidden_size=1,
    init_scale=0.0,
    scaler=true,
    MLE_tree_split=false
)
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

device = :cpu
loss = :mse # :mse :gaussian_mle :tweedie

learner = NeuroTabRegressor(
    arch;
    loss,
    nrounds=20,
    early_stopping_rounds=2,
    lr=1e-3,
    batchsize=1024,
    device
)

m = NeuroTabModels.fit(
    learner,
    dtrain;
    # deval,
    target_name,
    feature_names,
    print_every_n=5,
)

p_eval = m(deval; device=:cpu);
p_eval = p_eval[:, 1]
mse_eval = mean((p_eval .- deval.y_norm) .^ 2)
@info "MSE - deval" mse_eval

p_test = m(dtest; device=:cpu);
p_test = p_test[:, 1]
mse_test = mean((p_test .- dtest.y_norm) .^ 2) * std(df_tot.y_raw)^2
@info "MSE - dtest" mse_test
