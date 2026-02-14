using Random
using CUDA
using Flux, Optimisers
using Statistics: mean
import DifferentiationInterface as DI
import Mooncake as MC
# using Mooncake
# using Zygote
# using Enzyme

Random.seed!(123)

# m = Chain(BatchNorm(3), Dense(3, 1))
m = Chain(Dense(3, 3, relu), Dense(3, 1))
myloss(m, x, y) = mean((m(x) .- y) .^ 2)
x, y = randn(Float32, 3, 4), randn(Float32, 4)
myloss(m, x, y)
m, x, y = (m, x, y) .|> gpu

############
# Mooncake
############
cache = MC.prepare_gradient_cache(myloss, m, x, y; config=MC.Config(; friendly_tangents=true));
val, grads = MC.value_and_gradient!!(cache, myloss, m, x, y)
opts = Optimisers.setup(Adam(), m)
Optimisers.update!(opts, m, grads[2])

m, x, y = (m, x, y) .|> gpu

############
# Mooncake-GPU
############
mg, xg, yg = (m, x, y) .|> gpu
cache = MC.prepare_gradient_cache(myloss, mg, xg, yg; config=MC.Config(; friendly_tangents=true));
val, grads = MC.value_and_gradient!!(cache, myloss, mg, xg, yg)
opts = Optimisers.setup(Adam(), m)
Optimisers.update!(opts, m, grads[2])

############
# Mooncake DI
############
backend = DI.AutoMooncake(; config=MC.Config(; friendly_tangents=true))
typical_x, typical_y = similar(x), similar(y)
prep = DI.prepare_gradient(myloss, backend, m, DI.Constant(typical_x), DI.Constant(typical_y))
val, grads = DI.value_and_gradient(myloss, prep, backend, m, DI.Constant(x), DI.Constant(y))
opts = Optimisers.setup(Adam(), m)
Optimisers.update!(opts, m, grads)

############
# Zygote
############
backend = DI.AutoZygote()
typical_x, typical_y = similar(x), similar(y)
prep = DI.prepare_gradient(myloss, backend, m, DI.Constant(typical_x), DI.Constant(typical_y))
val, grads = DI.value_and_gradient(myloss, prep, backend, m, DI.Constant(x), DI.Constant(y))
opts = Optimisers.setup(Adam(), m)
Optimisers.update!(opts, m, grads)

############
# Enzyme
############
using Enzyme
backend = DI.AutoEnzyme()
typical_x, typical_y = similar(x), similar(y)
prep = DI.prepare_gradient(myloss, backend, m, DI.Constant(typical_x), DI.Constant(typical_y))
val, grads = DI.value_and_gradient(myloss, prep, backend, m, DI.Constant(x), DI.Constant(y))
opts = Optimisers.setup(Adam(), m)
Optimisers.update!(opts, m, grads)
