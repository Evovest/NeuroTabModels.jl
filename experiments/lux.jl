using NeuroTabModels
using Lux, LuxCore
using Random

using NeuroTabModels.Models.NeuroTrees: get_logits_mask, get_softplus_mask
using NNlib: softplus
using Reactant
using Enzyme
using Optimisers

rng = Random.Xoshiro(123)
nobs = 1000
nfeats = 10

# m = NeuroTabModels.Models.NeuroTrees.NeuroTree(nfeats => 1; depth=4, trees=64)
m = Chain(
    NeuroTabModels.Models.NeuroTrees.NeuroTree(nfeats => 1; depth=4, trees=64)
)
x = randn(rng, Float32, nfeats, nobs)
y = randn(rng, Float32, 1, nobs)
ps, st = LuxCore.setup(rng, m)
p, st = m(x, ps, st)

# Get the device determined by Lux
# Reactant.set_default_backend("gpu")
Reactant.set_default_backend("gpu")
dev = reactant_device()
# dev = gpu_device()
# dev = cpu_device()

# Parameter and State Variables
ps, st = Lux.setup(rng, m) |> dev

# Dummy Input
x = rand(rng, Float32, nfeats, nobs) |> dev

# Run the model
## We need to use @jit to compile and run the model with Reactant
@time p, st = @jit Lux.apply(m, x, ps, st)
# @time Lux.apply(m, x, ps, st)

## For best performance, first compile the model with Reactant and then run it
@time apply_compiled = @compile Lux.apply(m, x, ps, st)
@time apply_compiled(m, x, ps, st)

# Run the model
# Gradients
ts = Training.TrainState(m, ps, st, Adam(0.001f0))
gs, loss, stats, ts = Lux.Training.compute_gradients(
    AutoEnzyme(),
    MSELoss(),
    (x, y),
    ts
)

## Optimization
ts = Training.apply_gradients!(ts, gs) # or Training.apply_gradients (no `!` at the end)

# Both these steps can be combined into a single call (preferred approach)
@time gs, loss, stats, ts = Training.single_train_step!(
    AutoEnzyme(),
    MSELoss(),
    (x, y),
    ts
);
