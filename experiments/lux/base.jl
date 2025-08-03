using Lux, Random, Enzyme
using Functors, Optimisers
using Printf

# Reactant.set_default_backend("cpu")
rng = Random.default_rng()
Random.seed!(rng, 123)

dev = cpu_device()

model = Chain(
    Dense(2 => 4, gelu),
    Dense(4 => 4, gelu),
    Dense(4 => 2)
)
ps, st = Lux.setup(rng, model)

x_ra = [randn(Float32, 2, 32) for _ in 1:32]
y_ra = [xᵢ .^ 2 for xᵢ in x_ra]
ps_ra = ps |> dev
st_ra = st |> dev

dloader = DeviceIterator(dev, zip(x_ra, y_ra))

function train_model!(model, ps, st, dataloader)
    train_state = Training.TrainState(model, ps, st, Adam(0.001f0))

    for iteration in 1:1000
        for (i, (xᵢ, yᵢ)) in enumerate(dataloader)
            _, loss, _, train_state = Training.single_train_step!(
                AutoEnzyme(), MSELoss(), (xᵢ, yᵢ), train_state)
            if (iteration % 100 == 0 || iteration == 1) && i == 1
                @printf("Iter: [%4d/%4d]\tLoss: %.8f\n", iteration, 1000, loss)
            end
        end
    end

    return train_state
end

train_model!(model, ps, st, dloader)
