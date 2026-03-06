using NeuroTabModels
using DataFrames
using CategoricalArrays
using Lux

#################################
# vanilla DataFrame
#################################
nobs = 100
nfeats = 10
x = rand(nobs, nfeats);
df = DataFrame(x, :auto);
y = rand(nobs);
df.y = y;

target_name = "y"
feature_names = Symbol.(setdiff(names(df), [target_name]))
batchsize = 32

###################################
# CPU
###################################
device = :cpu
dtrain = NeuroTabModels.Data.get_df_loader_train(df; feature_names, target_name, batchsize, device)
for d in dtrain
    @info length(d)
    @info size(d[1]), size(d[2])
end

deval = NeuroTabModels.Data.get_df_loader_infer(df; feature_names, batchsize=32)
for d in deval
    @info size(d)
end

###################################
# LuxDevice
###################################
# dev = reactant_device()
dev = cpu_device()
# dev = gpu_device()
dtrain = NeuroTabModels.Data.get_df_loader_train(df; feature_names, target_name, batchsize) |> dev
for d in dtrain
    @info length(d)
    @info size(d[1])
    @info typeof(d[1])
end

###################################
# GPU
###################################
device = :gpu
dtrain = NeuroTabModels.Data.get_df_loader_train(df; feature_names, target_name, batchsize, device)
for d in dtrain
    @info length(d)
    @info size(d[1])
end

deval = NeuroTabModels.get_df_loader_infer(df; feature_names, batchsize=32)
for d in deval
    @info size(d)
end

###################################
# Categorical
###################################
target_name = "y"
feature_names = Symbol.(setdiff(names(df), [target_name]))
batchsize = 32
device = :gpu

x = rand(nobs, nfeats);
df = DataFrame(x, :auto);
df.y = categorical(rand(1:2, nobs));

dtrain = NeuroTabModels.get_df_loader_train(df; feature_names, target_name, batchsize, device)
for d in dtrain
    @info length(d)
    @info size(d[1])
    @info typeof(d[2])
end
