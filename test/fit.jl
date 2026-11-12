@testset "Regression - TabM" begin
    Random.seed!(123)
    X = randn(Float32, 1000, 10)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, 1000)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    target_name = "y"
    feature_names = setdiff(names(df), [target_name])
    train_indices = randperm(nrow(df))[1:800]
    dtrain, deval = df[train_indices, :], df[setdiff(1:nrow(df), train_indices), :]
    mse_baseline = mean((mean(dtrain.y) .- deval.y) .^ 2)

    # One arch type has to learn. The other two only have to fit and stay finite.
    for (arch_type, nrounds) in ((:tabm, 20), (:tabm_mini, 1), (:tabm_packed, 1))
        arch = NeuroTabModels.TabMConfig(; k=4, n_blocks=2, d_block=32, dropout=0.0, arch_type)
        learner = NeuroTabRegressor(arch; loss=:mse, nrounds, lr=1e-2, device=:cpu, backend=:zygote)
        m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, deval)
        p = m(deval)
        @test size(p, 1) == nrow(deval)
        @test all(isfinite, p)
        arch_type == :tabm && @test mean((p .- deval.y) .^ 2) < mse_baseline
    end
end

@testset "Classification - NeuroTree" begin
    Random.seed!(123)
    X, y = @load_crabs
    df = DataFrame(X)
    df[!, :class] = y
    target_name = "class"
    feature_names = setdiff(names(df), [target_name])
    transform!(df, feature_names .=> (x -> (x .- mean(x)) ./ std(x)); renamecols=false)

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    learner = NeuroTabClassifier(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(:depth => 4),
        embedding_config=Dict(:embedding_type => :batchnorm),
        nrounds=200,
        early_stopping_rounds=5,
        lr=3e-2,
    )

    m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names)
    # Predictions depend on the number of samples in the dataset
    ptrain = [argmax(x) for x in eachrow(m(dtrain))]
    peval = [argmax(x) for x in eachrow(m(deval))]
    @test mean(ptrain .== levelcode.(dtrain.class)) >= 0.95
    @test mean(peval .== levelcode.(deval.class)) >= 0.95
end

@testset "Regression - grouped inference row order" begin
    Random.seed!(123)
    nobs = 200
    X = randn(Float32, nobs, 4)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, nobs)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    df[!, :grp] = repeat(1:10, inner=20)
    target_name = "y"
    feature_names = setdiff(names(df), [target_name, "grp"])

    dtrain = sort(df[1:160, :], :grp)
    # the caller's frame is deliberately not in group order
    deval = df[161:end, :][randperm(40), :]

    arch = NeuroTabModels.MLPConfig(; hidden_size=32)
    learner = NeuroTabRegressor(arch; loss=:mse, nrounds=40, lr=1e-2, batchsize=32)
    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, group_name="grp")

    p = m(deval)
    @test size(p, 1) == nrow(deval)
    # predictions must vary, or a misalignment would compare equal anyway
    @test std(p) > 0.5
    # one row per call cannot be reordered by grouping, so it pins each row's own prediction
    p_row = [m(deval[i:i, :])[1] for i in 1:nrow(deval)]
    @test maximum(abs.(p .- p_row)) < 1e-5
end

@testset "Regression - seed makes a fit reproducible" begin
    Random.seed!(123)
    nobs = 200
    X = randn(Float32, nobs, 4)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, nobs)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    df[!, :grp] = repeat(1:10, inner=20)
    target_name = "y"
    feature_names = setdiff(names(df), [target_name, "grp"])

    arch = NeuroTabModels.MLPConfig(; hidden_size=16)
    function fit_predict(seed, global_seed; group_name=nothing)
        # a different global stream must not change the fit, only `seed` may
        Random.seed!(global_seed)
        learner = NeuroTabRegressor(arch; loss=:mse, nrounds=5, lr=1e-2, batchsize=32, seed)
        m = NeuroTabModels.fit(learner, df; target_name, feature_names, group_name)
        return m(df)
    end

    for group_name in (nothing, "grp")
        p = fit_predict(123, 1; group_name)
        @test p == fit_predict(123, 2; group_name)
        @test p != fit_predict(124, 1; group_name)
    end
end

@testset "Classification - grouped" begin
    Random.seed!(123)
    nobs = 360
    X = randn(Float32, nobs, 4)
    score = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.2f0 .* randn(Float32, nobs)
    df = DataFrame(X, :auto)
    df[!, :class] = categorical(ifelse.(score .> 0.5, "hi", ifelse.(score .< -0.5, "lo", "mid")))
    # unequal group sizes, so the grouped loader pads
    df[!, :grp] = rand(1:12, nobs)
    target_name = "class"
    feature_names = setdiff(names(df), [target_name, "grp"])
    dtrain = df[1:280, :]
    deval = df[281:end, :]

    arch = NeuroTabModels.MLPConfig(; hidden_size=32)
    learner = NeuroTabClassifier(arch; nrounds=40, lr=1e-2, batchsize=32)
    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, deval, group_name="grp")

    p = m(deval)
    @test size(p) == (nrow(deval), 3)
    @test maximum(abs.(sum(p; dims=2) .- 1)) < 1e-5
    @test mean(argmax.(eachrow(p)) .== levelcode.(deval.class)) > 0.6
    # the eval metric is scored on the padded groups, so it must match the rows alone
    mlogloss = mean(-log(p[i, levelcode(deval.class[i])]) for i in 1:nrow(deval))
    @test m.info[:logger][:best_metric] ≈ mlogloss rtol = 1e-4
end

@testset "Backend/device - reactant is a backend" begin
    @test_throws ErrorException NeuroTabRegressor(; backend=:zygote, device=:reactant)
    @test_throws ErrorException NeuroTabClassifier(; backend=:zygote, device=:reactant)
end

@testset "Backend/device - Regression ($backend, $device)" for (backend, device) in
    [(:enzyme, :cpu), (:zygote, :cpu), (:reactant, :cpu)]
    Random.seed!(123)
    X = randn(Float32, 1000, 10)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, 1000)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    target_name = "y"
    feature_names = setdiff(names(df), [target_name])

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    learner = NeuroTabRegressor(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(:depth => 3),
        loss=:mse,
        nrounds=20,
        early_stopping_rounds=2,
        lr=1e-1,
        backend,
        device,
    )

    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, deval)

    p = m(deval)
    @test size(p, 1) == nrow(deval)
    @test !any(isnan, p)
    mse_model = mean((p .- deval.y) .^ 2)
    mse_baseline = mean((mean(dtrain.y) .- deval.y) .^ 2)
    @test mse_model < mse_baseline
end

@testset "Regression - Poisson" begin
    Random.seed!(123)
    X = randn(Float32, 1000, 10)
    μ = exp.(0.5f0 .* X[:, 1] .- 0.3f0 .* X[:, 2])
    # Knuth's sampler: count uniform draws until their product falls below exp(-μ).
    rpois(λ) = (k = 0; p = rand(); while p > exp(-λ); k += 1; p *= rand(); end; k)
    df = DataFrame(X, :auto)
    df[!, :y] = Float32.(rpois.(μ))
    target_name = "y"
    feature_names = setdiff(names(df), [target_name])

    train_indices = randperm(nrow(df))[1:800]
    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    learner = NeuroTabRegressor(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(:depth => 3),
        loss=:poisson,
        nrounds=20,
        early_stopping_rounds=2,
        lr=1e-1,
    )
    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, deval, print_every_n=5)

    p = m(deval)
    @test size(p, 1) == nrow(deval)
    dev(μ̂, y) = 2 * ((y > 0 ? y * log(y / μ̂) : 0.0) + μ̂ - y)
    dev_model = mean(dev.(p, deval.y))
    dev_baseline = mean(dev.(mean(dtrain.y), deval.y))
    @test dev_model < dev_baseline
end

@testset "Pearson from a flat start" begin
    # Zero leaves make every prediction equal at iteration 0. A NaN first eval would become the
    # best metric, and early stopping would then fire with best_iter 0.
    Random.seed!(123)
    df = DataFrame(randn(Float32, 40 * 16, 4), :auto)
    df[!, :y] = df.x1 .+ 0.15f0 .* randn(Float32, nrow(df))
    df[!, :grp] = repeat(1:40, inner=16)
    dtrain = df[df.grp.<=32, :]
    deval = df[df.grp.>32, :]
    feature_names = ["x1", "x2", "x3", "x4"]

    arch = NeuroTabModels.NeuroTreeConfig(; depth=3, ntrees=8, stack_size=1)
    for backend in (:zygote, :reactant)
        learner = NeuroTabRegressor(
            arch; loss=:pearson, metric=:pearson, nrounds=12, early_stopping_rounds=4, lr=1e-2,
            batchsize=0, backend, device=:cpu,
        )
        m = NeuroTabModels.fit(learner, dtrain; target_name="y", feature_names, deval, group_name="grp")
        metrics = m.info[:logger][:metrics][:metric]
        @test all(isfinite, metrics)
        @test first(metrics) ≈ 0 atol = 1e-5
        @test m.info[:logger][:best_iter] > 0
        @test maximum(metrics) > 0.8
        @test m.info[:group_name] === :grp
    end

    # One group with a flat target scores 0 and must not turn the whole metric NaN.
    deval_flat = copy(deval)
    deval_flat[deval_flat.grp.==40, :y] .= 1.0f0
    learner = NeuroTabRegressor(
        NeuroTabModels.MLPConfig(; hidden_size=16, stack_size=1, dropout=0.0);
        loss=:mse, metric=:pearson, nrounds=2, lr=1e-2, batchsize=0, backend=:zygote, device=:cpu,
    )
    m = NeuroTabModels.fit(learner, dtrain; target_name="y", feature_names, deval=deval_flat, group_name="grp")
    @test all(isfinite, m.info[:logger][:metrics][:metric])
end

@testset "eval_group_name is independent of group_name" begin
    Random.seed!(123)
    df = DataFrame(randn(Float32, 64, 4), :auto)
    df.y = df.x1
    df.grp = repeat(1:8, inner=8)
    learner = NeuroTabRegressor(
        NeuroTabModels.MLPConfig(; hidden_size=16); loss=:mse, metric=:pearson,
        nrounds=1, batchsize=32, backend=:zygote, device=:cpu,
    )
    m = NeuroTabModels.fit(
        learner, df; target_name="y", feature_names=["x1", "x2", "x3", "x4"], deval=df, eval_group_name="grp",
    )
    @test isnothing(m.info[:group_name])
    @test m.info[:eval_group_name] === :grp
    @test all(isfinite, m.info[:logger][:metrics][:metric])
end

@testset "Regression - Huber" begin
    Random.seed!(123)
    X = randn(Float32, 800, 6)
    y = X[:, 1] .- 0.4f0 .* X[:, 2] .+ 0.05f0 .* randn(Float32, 800)
    y[1:8] .= 20.0f0
    df = DataFrame(X, :auto)
    df[!, :y] = y
    feature_names = ["x1", "x2", "x3", "x4", "x5", "x6"]
    dtrain, deval = df[1:640, :], df[641:end, :]
    arch = NeuroTabModels.MLPConfig(; hidden_size=32, stack_size=1, dropout=0.0)
    learner = NeuroTabRegressor(
        arch; loss=:huber, metric=:huber, nrounds=25, early_stopping_rounds=25,
        lr=1e-2, batchsize=64, backend=:zygote, device=:cpu,
    )
    m = NeuroTabModels.fit(learner, dtrain; target_name="y", feature_names, deval, print_every_n=25, verbosity=0)
    metrics = m.info[:logger][:metrics][:metric]
    @test all(isfinite, metrics)
    @test last(metrics) < first(metrics)
    p = m(deval)
    @test size(p) == (nrow(deval),)
    @test !any(isnan, p)
    @test cor(p, deval.y) > 0.7
end

@testset "Unknown loss and metric are rejected" begin
    @test_throws "Unknown loss" NeuroTabModels.Losses.LossType(:not_a_loss)
    @test_throws "not supported" NeuroTabRegressor(NeuroTabModels.MLPConfig(); loss=:not_a_loss)
    @test_throws "Invalid metric" NeuroTabRegressor(NeuroTabModels.MLPConfig(); metric=:not_a_metric)
end

@testset "Grouped architectures fit and stay finite" begin
    Random.seed!(123)
    df = DataFrame(randn(Float32, 64, 4), :auto)
    df.y = df.x1
    df.grp = repeat(1:8, inner=8)
    feature_names = ["x1", "x2", "x3", "x4"]
    archs = [
        NeuroTabModels.MLPConfig(; hidden_size=16),
        NeuroTabModels.ResNetConfig(; hidden_size=16, stack_size=1),
        NeuroTabModels.MLPAttnConfig(; hidden_size=16, nheads=2, stack_size=1),
        NeuroTabModels.NeuroTreeAttnConfig(; hidden_size=16, nheads=2, depth=2, ntrees=4),
    ]
    for arch in archs
        learner = NeuroTabRegressor(arch; loss=:mse, nrounds=1, batchsize=32, device=:cpu, backend=:zygote)
        m = NeuroTabModels.fit(learner, df; target_name="y", feature_names, group_name="grp")
        p = m(df)
        @test length(p) == nrow(df)
        @test all(isfinite, p)
    end
end

@testset "Weights and offsets reach the fit" begin
    # Odd rows follow y = 2 x1 and even rows y = -x1, so weighting odd rows 9 to 1
    # gives a least-squares slope of 1.7 against 0.5 unweighted.
    Random.seed!(123)
    nobs = 400
    X = randn(Float32, nobs, 4)
    odd = isodd.(1:nobs)
    y = ifelse.(odd, 2 .* X[:, 1], .-X[:, 1]) .+ 0.1f0 .* randn(Float32, nobs)
    d = DataFrame(X, :auto)
    d[!, :y] = y
    d[!, :w] = Float32.(ifelse.(odd, 9, 1))
    d[!, :grp] = repeat(1:20, inner=20)
    feature_names = ["x1", "x2", "x3", "x4"]
    arch = NeuroTabModels.MLPConfig(; hidden_size=32)
    learner = NeuroTabRegressor(arch; loss=:mse, nrounds=40, lr=1e-2, batchsize=32)
    x1 = Float64.(X[:, 1])
    slope(pred, x) = sum((pred .- mean(pred)) .* (x .- mean(x))) / sum((x .- mean(x)) .^ 2)
    pw = Float64.(NeuroTabModels.fit(learner, d; target_name="y", feature_names, group_name="grp", weight_name="w")(d))
    pu = Float64.(NeuroTabModels.fit(learner, d; target_name="y", feature_names, group_name="grp")(d))
    @test slope(pw, x1) > 1.2
    @test slope(pw, x1) > slope(pu, x1) + 0.5

    # The target is x1 + 2 x2 and the offset carries 2 x2, so with it the model learns x1 alone.
    d[!, :off] = 2 .* X[:, 2]
    d[!, :y] = X[:, 1] .+ d.off .+ 0.1f0 .* randn(Float32, nobs)
    learner = NeuroTabRegressor(arch; loss=:mse, metric=:pearson, nrounds=40, lr=1e-2, batchsize=32, scale_target=false)
    x2 = Float64.(X[:, 2])
    mo = NeuroTabModels.fit(learner, d; target_name="y", feature_names, group_name="grp", offset_name="off", deval=d)
    po = Float64.(mo(d))
    pn = Float64.(NeuroTabModels.fit(learner, d; target_name="y", feature_names, group_name="grp")(d))
    @test abs(slope(po, x2)) < 0.5
    @test slope(pn, x2) > 1.5
    groups = ((20 * (i - 1) + 1):(20 * i) for i in 1:20)
    r = [cor(po[g] .+ d.off[g], Float64.(d.y[g])) for g in groups]
    @test mo.info[:logger][:best_metric] ≈ mean(r) rtol = 1e-4
end
