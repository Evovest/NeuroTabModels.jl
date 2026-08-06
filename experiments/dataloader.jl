using NeuroTabModels
using DataFrames
using CategoricalArrays
using DataFrames
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
dev = :cpu
dtrain = NeuroTabModels.Data.get_df_loader_train(df; feature_names, target_name, batchsize)
x, y = first(dtrain)
for d in dtrain
    @info length(d)
    @info size(d[1]), size(d[2])
end

deval = NeuroTabModels.Data.get_df_loader_infer(df; feature_names, batchsize=32)
for d in deval
    @info size(d)
end

###################################
# grouped-DF - CPU
###################################
df.grp = rand(1:round(Int, nrow(df) / 32), nrow(df))
dev = :cpu
dfg = groupby(df, :grp)
dtrain = NeuroTabModels.Data.get_df_loader_train(dfg; feature_names, target_name, batchsize)
for d in dtrain
    @info length(d)
    @info size(d[1]), size(d[2])
    @info sum(d[2] .!= 0)
    @info sum(d[3])
end
x, y, w = first(dtrain)

###################################
# LuxDevice
###################################
dev = reactant_device()
# dev = cpu_device()
dtrain = NeuroTabModels.Data.get_df_loader_train(df; feature_names, target_name, batchsize) |> dev
for d in dtrain
    @info length(d)
    @info size(d[1])
    @info typeof(d[1])
end

###################################
# GPU
###################################
dev = gpu_device()
dtrain = NeuroTabModels.Data.get_df_loader_train(df; feature_names, target_name, batchsize) |> dev
for d in dtrain
    @info length(d)
    @info size(d[1])
    @info typeof(d[1])
end

deval = NeuroTabModels.Data.get_df_loader_infer(df; feature_names, batchsize=32)
for d in deval
    @info size(d)
end

###################################
# Categorical
###################################
target_name = "y"
feature_names = Symbol.(setdiff(names(df), [target_name]))
batchsize = 32
dev = cpu_device()

x = rand(nobs, nfeats);
df = DataFrame(x, :auto);
df.y = categorical(rand(1:2, nobs));

dtrain = NeuroTabModels.Data.get_df_loader_train(df; feature_names, target_name, batchsize) |> dev
for d in dtrain
    @info length(d)
    @info size(d[1])
    @info typeof(d[2])
end
