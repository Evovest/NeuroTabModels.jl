module NeuroTabModels

using Random

export NeuroTabRegressor, NeuroTabClassifier, NeuroTabModel

include("utils.jl")
include("learners.jl")
using .Learners
include("data.jl")
# using .Data
include("losses.jl")
# using .Losses
include("metrics.jl")
# using .Metrics
include("models/models.jl")
using .Models
include("infer.jl")
using .Infer
include("Fit/fit.jl")
using .Fit
include("MLJ.jl")

end # module
