using Random
using CSV
using DataFrames
using StatsBase
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
transform!(df_tot, "y_raw" => (x -> (x .- mean(x)) ./ std(x)) => "y_norm")
select!(df_tot, Not("Column1"))
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
    actA=:identity,
    depth=4,
    ntrees=32,
    stack_size=1,
    hidden_size=1,
    init_scale=0.1,
    MLE_tree_split=true
)

device = :gpu
learner = NeuroTabRegressor(
    arch;
    loss=:gaussian_mle,
    nrounds=200,
    early_stopping_rounds=2,
    lr=1e-3,
    batchsize=2048,
    device
)

m = NeuroTabModels.fit(
    learner,
    dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n=5,
)

p_test = m(dtest; device);
mse_test = mean((p_test[:, 1] .- dtest.y_norm) .^ 2) * std(df_tot.y_raw)^2
@info "MSE - dtest" mse_test

# NeuroTrees.save(m, "data/YEAR/model_1_mse.bson")
# m2 = NeuroTrees.load("data/YEAR/model_1_mse.bson")[:model]
# NeuroTrees.save(m, "data/YEAR/model_1.bson")
# @code_warntype Modeler.Models.NeuroTrees.infer(m, dinfer)

# @time pred1 = NeuroTrees.infer(m, dinfer);
# @time pred2 = NeuroTrees.infer(m, dinfer);

# @btime pred1 = NeuroTrees.infer($m, $dinfer);
# @btime pred2 = NeuroTrees.infer($m, $dinfer);