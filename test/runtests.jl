using Test
using NeuroTabModels
using Lux
using Tables
using DataFrames
using Statistics: mean, std, cor
using CategoricalArrays
using Random
using MLJBase
using MLJTestInterface
using Enzyme
using Reactant
using Zygote

include("data.jl")
include("models.jl")
include("losses.jl")
include("fit.jl")
include("embedding.jl")
include("MLJ.jl")
