using Random
using CairoMakie
using DataFrames
using NeuroTabModels
using Reactant

device = :cpu
assets = joinpath(@__DIR__, "..", "docs", "src", "assets")

nobs = 10_000
Random.seed!(123)
x_num = rand(nobs) .* 5

function noisy_sin(x)
    y = clamp.(sin.(x) .* 0.5 .+ 0.5, eps(Float64), 1 - eps(Float64))
    logits = log.(y ./ (1 .- y)) .+ randn(length(x)) .* 0.5
    return 1 ./ (1 .+ exp.(-logits))
end

dtot = DataFrame(x_num=x_num, y=noisy_sin(x_num), y2=noisy_sin(2x_num .- 2))
is = randperm(nobs)
ntrain = floor(Int, 0.8 * nobs)
dtrain = dtot[is[1:ntrain], :]
deval = dtot[is[(ntrain+1):end], :]

train_kw = (
    nrounds=400,
    lr=2e-2,
    seed=123,
    early_stopping_rounds=5,
    backend=:reactant,
    device,
)

architectures = [
    "neurotree" => NeuroTabModels.NeuroTreeConfig(; depth=4, ntrees=16, k=4),
    # Add "mlp" => NeuroTabModels.MLPConfig(...) or "tabm" => NeuroTabModels.TabMConfig(...) here.
]

losses = [:mse, :mae, :logloss, :poisson, :tweedie, :gaussian_mle]
# losses = [:mse]

function fit_and_predict(arch, loss)
    config = NeuroTabRegressor(arch; train_kw..., loss)
    @time model = NeuroTabModels.fit(
        config,
        dtrain;
        feature_names=["x_num"],
        target_name=["y", "y2"],
        deval,
        print_every_n=5,
        # verbosity=0,
    )
    @time pred = model(dtrain; device)
    return pred
end

function plot_multi(pred; name, arch_name, nparams)
    size(pred) == (nrow(dtrain), 2 * nparams) ||
        error("Expected $(2 * nparams) prediction columns for $name, got size $(size(pred)).")
    x = dtrain.x_num
    perm = sortperm(x)
    f = Figure()
    ax = Axis(f[1, 1], xlabel="feature", ylabel="target")
    for (i, (target, color)) in enumerate((("y", "#26a671"), ("y2", "#e5616c")))
        scatter!(ax, x[perm], dtrain[perm, target]; markersize=2, color, label=target)
        col = (i - 1) * nparams + 1
        mu = pred[perm, col]
        if nparams == 2
            sigma = pred[perm, col+1]
            band!(ax, x[perm], mu .- sigma, mu .+ sigma; color=(color, 0.2))
        end
        lines!(ax, x[perm], mu; linewidth=2, color)
    end
    Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
    save(joinpath(assets, "multi-target-$name-$arch_name-$device.png"), f)
    return f
end

for (arch_name, arch) in architectures
    for loss in losses
        pred = fit_and_predict(arch, loss)
        plot_multi(pred; name=loss, arch_name, nparams=loss == :gaussian_mle ? 2 : 1)
    end
end
