using Test
using NeuroTabModels
using Tables
using DataFrames
using Statistics: mean
using CategoricalArrays
using StatsBase: sample
using Random
using MLJBase
using MLJTestInterface

include("core.jl")
include("embedding.jl")
include("MLJ.jl")
