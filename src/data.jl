module Data

export get_df_loader_train, get_df_loader_infer, feature_matrix

import Base: length, getindex
import MLUtils: DataLoader, getobs, numobs
import Random: default_rng

using DataFrames
using CategoricalArrays

"""
    feature_matrix(df, features)

Model input for the `features` columns of `df`. A vector of names gives a `(nfeats, nobs)`
`Float32` matrix. A NamedTuple gives one matrix per slot, where a slot of `name => levels`
pairs gives `Int32` level codes: the position of the value in the training levels, or `0`
for a level not seen at fit time. Any other `missing` becomes `NaN`.
"""
feature_matrix(df::AbstractDataFrame, features::NamedTuple) = map(f -> feature_matrix(df, f), features)
function feature_matrix(df::AbstractDataFrame, names::AbstractVector)
    return stack(Float32.(coalesce.(df[!, f], NaN32)) for f in names; dims=1)
end
function feature_matrix(df::AbstractDataFrame, levels::AbstractVector{<:Pair})
    return stack(Int32.(something.(indexin(unwrap.(df[!, f]), l), 0)) for (f, l) in levels; dims=1)
end

# Pads a group's columns with zeros up to the `n` observations of the group buffer.
_pad(x::AbstractMatrix, n) = hcat(x, zeros(eltype(x), size(x, 1), n - size(x, 2)))
_pad(x::NamedTuple, n) = map(m -> _pad(m, n), x)

"""
    ContainerTrain
"""
struct ContainerTrain{A,B,C,D}
    x::A
    y::B
    w::C
    offset::D
end

length(data::ContainerTrain) = numobs(data.x)
length(data::ContainerTrain{<:Vector}) = length(data.x)

function getindex(data::ContainerTrain{A,B,C,D}, idx::AbstractVector) where {A,B,C<:Nothing,D<:Nothing}
    x = getobs(data.x, idx)
    y = data.y[:, idx]
    return (x, y)
end
function getindex(data::ContainerTrain{A,B,C,D}, idx::AbstractVector) where {A,B,C<:AbstractVector,D<:Nothing}
    x = getobs(data.x, idx)
    y = data.y[:, idx]
    w = data.w[idx]
    return (x, y, w)
end
function getindex(data::ContainerTrain{A,B,C,D}, idx::AbstractVector) where {A,B,C<:AbstractVector,D<:AbstractVector}
    x = getobs(data.x, idx)
    y = data.y[:, idx]
    w = data.w[idx]
    offset = data.offset[idx]
    return (x, y, w, offset)
end
function getindex(data::ContainerTrain{A,B,C,D}, idx::AbstractVector) where {A,B,C<:AbstractVector,D<:AbstractMatrix}
    x = getobs(data.x, idx)
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
    x = feature_matrix(df, feature_names)

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
    bs = maximum(dfg.ends .- dfg.starts) + 1
    # bs=2048 # FIXME: stress test for reactant memory issue
    @info "group train bs: $bs"

    x = [_pad(feature_matrix(dfg[i], feature_names), bs) for i in 1:n]
    ntargets = length(vcat(target_name))
    y = [zeros(Float32, ntargets, 1, bs) for _ in 1:n]
    mask = [zeros(Bool, bs) for _ in 1:n]
    w = [zeros(Float32, 1, 1, bs) for _ in 1:n]
    # one row per offset column, as in the ungrouped (K, N) offset, and zero on pads
    offset_names = offset_name isa Union{String,Symbol} ? [offset_name] : offset_name
    offset = isnothing(offset_name) ? nothing : [zeros(Float32, length(offset_names), 1, bs) for _ in 1:n]

    for i in 1:n
        df = dfg[i]
        mask[i][1:nrow(df)] .= true
        if target_name isa AbstractVector
            target = Matrix(df[:, target_name])'
        else
            target = df[!, target_name]
            if eltype(target) <: CategoricalValue
                target = CategoricalArrays.levelcode.(target)
            end
            target = reshape(target, 1, :)
        end
        if isnothing(scalers)
            y[i][:, 1, 1:nrow(df)] .= target
        else
            y[i][:, 1, 1:nrow(df)] .= (target .- scalers[:mu]) ./ scalers[:sigma]
        end
        if isnothing(weight_name)
            w[i][1, 1, 1:nrow(df)] .= 1.0
        else
            # each step is normalised by its group's weight sum
            wi = Float32.(df[!, weight_name])
            all(v -> isfinite(v) && v >= 0, wi) && sum(wi) > 0 ||
                error("Weights in `$weight_name` must be non-negative and finite, with a positive sum per group.")
            w[i][1, 1, 1:nrow(df)] .= wi
        end
        if !isnothing(offset)
            offset[i][:, 1, 1:nrow(df)] .= Matrix(df[:, offset_names])'
        end
    end

    container = ContainerTrain(collect(zip(x, mask)), y, w, offset)
    dtrain = DataLoader(container; shuffle, batchsize=0, partial=false, parallel=false, rng)
    return dtrain
end

"""
    ContainerInfer
"""
struct ContainerInfer{A}
    x::A
end
length(data::ContainerInfer) = numobs(data.x)

function getindex(data::ContainerInfer, idx::AbstractVector)
    x = getobs(data.x, idx)
    return x
end

function get_df_loader_infer(df::AbstractDataFrame; feature_names, batchsize)
    x = feature_matrix(df, feature_names)

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
    bs = maximum(dfg.ends .- dfg.starts) + 1
    @info "group infer bs: $bs"

    x = [_pad(feature_matrix(dfg[i], feature_names), bs) for i in 1:n]
    mask = [zeros(Bool, bs) for _ in 1:n]

    for i in 1:n
        df = dfg[i]
        mask[i][1:nrow(df)] .= true
    end

    container = ContainerInferGrp(x, mask)
    dinfer = DataLoader(container; shuffle=false, batchsize=0, partial=false, parallel=false)
    return dinfer
end

end #module