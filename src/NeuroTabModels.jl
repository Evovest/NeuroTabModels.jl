module NeuroTabModels

using Random

export NeuroTabRegressor, NeuroTabClassifier, NeuroTabModel

include("data.jl")
include("losses.jl")
include("metrics.jl")
include("models/models.jl")
using .Models
include("learners.jl")
using .Learners
include("infer.jl")
using .Infer
include("Fit/fit.jl")
using .Fit
include("MLJ.jl")

end # module
