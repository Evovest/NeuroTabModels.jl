module Data

export get_df_loader_train, get_df_loader_infer

import Base: length, getindex
import MLUtils: DataLoader
import Random: default_rng

using DataFrames
using CategoricalArrays

"""
    ContainerTrain
"""
struct ContainerTrain{A,B,C,D}
    x::A
    y::B
    w::C
    offset::D
end

length(data::ContainerTrain) = size(data.x, 2)
length(data::ContainerTrain{<:Vector}) = length(data.x)

function getindex(data::ContainerTrain{A,B,C,D}, idx::AbstractVector) where {A,B,C<:Nothing,D<:Nothing}
    x = data.x[:, idx]
    y = data.y[:, idx]
    return (x, y)
end
function getindex(data::ContainerTrain{A,B,C,D}, idx::AbstractVector) where {A,B,C<:AbstractVector,D<:Nothing}
    x = data.x[:, idx]
    y = data.y[:, idx]
    w = data.w[idx]
    return (x, y, w)
end
function getindex(data::ContainerTrain{A,B,C,D}, idx::AbstractVector) where {A,B,C<:AbstractVector,D<:AbstractVector}
    x = data.x[:, idx]
    y = data.y[:, idx]
    w = data.w[idx]
    offset = data.offset[idx]
    return (x, y, w, offset)
end
function getindex(data::ContainerTrain{A,B,C,D}, idx::AbstractVector) where {A,B,C<:AbstractVector,D<:AbstractMatrix}
    x = data.x[:, idx]
    y = data.y[:, idx]
    w = data.w[idx]
    offset = data.offset[:, idx]
    return (x, y, w, offset)
end

function get_df_loader_train(
    df::AbstractDataFrame;
    feature_names,
    target_name,
    weight_name=nothing,
    offset_name=nothing,
    batchsize,
    scalers=nothing,
    shuffle=true,
    rng=default_rng(),
)
    feature_names = Symbol.(feature_names)
    x = Matrix{Float32}(Matrix{Float32}(select(df, feature_names))')

    # `(T, N)`: one row per target; per-target scalers broadcast down the rows
    if target_name isa AbstractVector
        y = Matrix{Float32}(Matrix{Float32}(select(df, target_name))')
    elseif eltype(df[!, target_name]) <: CategoricalValue
        y = reshape(UInt32.(CategoricalArrays.levelcode.(df[!, target_name])), 1, :)
    else
        y = reshape(Float32.(df[!, target_name]), 1, :)
    end
    if !isnothing(scalers)
        y .= (y .- scalers[:mu]) ./ scalers[:sigma]
    end

    # batches carrying an offset are laid out as (x, y, w, offset), so unit weights stand in when none are given
    w = if !isnothing(weight_name)
        Float32.(df[!, weight_name])
    elseif !isnothing(offset_name)
        ones(Float32, size(y, 2))
    else
        nothing
    end

    offset = if isnothing(offset_name)
        nothing
    else
        if offset_name isa Union{String,Symbol}
            Float32.(df[!, offset_name])
        else
            Matrix{Float32}(Matrix{Float32}(df[!, offset_name])')
        end
    end

    container = ContainerTrain(x, y, w, offset)
    batchsize = min(batchsize, length(container))
    dtrain = DataLoader(container; shuffle, batchsize, partial=false, parallel=false, rng)
    return dtrain
end

# for GroupedDataFrame
function getindex(data::ContainerTrain{A,B,C,D}, idx::Integer) where {A<:Vector,B<:Vector,C<:Vector,D<:Nothing}
    x = data.x[idx]
    y = data.y[idx]
    w = data.w[idx]
    return (x, y, w)
end
function getindex(data::ContainerTrain{A,B,C,D}, idx::Integer) where {A<:Vector,B<:Vector,C<:Vector,D<:Vector}
    x = data.x[idx]
    y = data.y[idx]
    w = data.w[idx]
    offset = data.offset[idx]
    return (x, y, w, offset)
end

function get_df_loader_train(
    dfg::GroupedDataFrame;
    feature_names,
    target_name,
    weight_name=nothing,
    offset_name=nothing,
    batchsize=0,
    scalers=nothing,
    shuffle=true,
    rng=default_rng(),
)
    n = length(dfg)
    nfeats = length(feature_names)
    bs = maximum(dfg.ends .- dfg.starts) + 1
    # bs=2048 # FIXME: stress test for reactant memory issue
    @info "group train bs: $bs"

    x = [zeros(Float32, nfeats, bs) for _ in 1:n]
    # one row per target, as in the ungrouped `(T, N)` target, and zero on pads
    ntargets = target_name isa AbstractVector ? length(target_name) : 1
    y = [zeros(Float32, ntargets, 1, bs) for _ in 1:n]
    w = [zeros(Float32, 1, 1, bs) for _ in 1:n]
    # one row per offset column, as in the ungrouped (K, N) offset, and zero on pads
    offset_names = offset_name isa Union{String,Symbol} ? [offset_name] : offset_name
    offset = isnothing(offset_name) ? nothing : [zeros(Float32, length(offset_names), 1, bs) for _ in 1:n]

    for i in 1:n
        df = dfg[i]
        x[i][:, 1:nrow(df)] .= Matrix(df[:, feature_names])'
        if target_name isa AbstractVector
            target = Matrix(df[:, target_name])'
        else
            target = df[!, target_name]
            if eltype(target) <: CategoricalValue
                target = CategoricalArrays.levelcode.(target)
            end
            target = reshape(target, 1, :)
        end
        # per-target scalers are `(T,)` vectors and broadcast down the rows
        if isnothing(scalers)
            y[i][:, 1, 1:nrow(df)] .= target
        else
            y[i][:, 1, 1:nrow(df)] .= (target .- scalers[:mu]) ./ scalers[:sigma]
        end
        if isnothing(weight_name)
            w[i][1, 1, 1:nrow(df)] .= 1.0
        else
            # a weight of zero marks a padded slot, so real rows need a positive one
            wi = Float32.(df[!, weight_name])
            all(v -> isfinite(v) && v > 0, wi) ||
                error("Weights in `$weight_name` must be positive and finite.")
            w[i][1, 1, 1:nrow(df)] .= wi
        end
        if !isnothing(offset)
            offset[i][:, 1, 1:nrow(df)] .= Matrix(df[:, offset_names])'
        end
    end

    container = ContainerTrain(x, y, w, offset)
    dtrain = DataLoader(container; shuffle, batchsize=0, partial=false, parallel=false, rng)
    return dtrain
end

"""
    ContainerInfer
"""
struct ContainerInfer{A<:AbstractMatrix}
    x::A
end
length(data::ContainerInfer) = size(data.x, 2)

function getindex(data::ContainerInfer, idx::AbstractVector)
    x = data.x[:, idx]
    return x
end

function get_df_loader_infer(df::AbstractDataFrame; feature_names, batchsize)
    feature_names = Symbol.(feature_names)
    x = Matrix{Float32}(Matrix{Float32}(select(df, feature_names))')

    container = ContainerInfer(x)
    batchsize = min(batchsize, length(container))
    dinfer = DataLoader(container; shuffle=false, batchsize, partial=true, parallel=false)
    return dinfer
end

"""
    ContainerInferGrp
"""
struct ContainerInferGrp{A<:AbstractVector,B<:AbstractVector}
    x::A
    mask::B
end
length(data::ContainerInferGrp) = length(data.x)

# for GroupedDataFrame
function getindex(data::ContainerInferGrp, idx::Int)
    x = data.x[idx]
    mask = data.mask[idx]
    return (x, mask)
end
# function getindex(data::ContainerInferGrp, idx::AbstractVector)
#     x = data.x[first(idx)]
#     mask = data.mask[first(idx)]
#     return (x, mask)
# end

function get_df_loader_infer(dfg::GroupedDataFrame; feature_names, batchsize=0)
    n = length(dfg)
    nfeats = length(feature_names)
    bs = maximum(dfg.ends .- dfg.starts) + 1
    @info "group infer bs: $bs"

    x = [zeros(Float32, nfeats, bs) for _ in 1:n]
    mask = [zeros(Bool, bs) for _ in 1:n]

    for i in 1:n
        df = dfg[i]
        x[i][:, 1:nrow(df)] .= Matrix(df[!, feature_names])'
        mask[i][1:nrow(df)] .= true
    end

    container = ContainerInferGrp(x, mask)
    dinfer = DataLoader(container; shuffle=false, batchsize=0, partial=false, parallel=false)
    return dinfer
end

end #module