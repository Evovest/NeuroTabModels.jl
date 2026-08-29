@testset "Core - data iterators" begin end

@testset "Core - internals test" begin
    learner = NeuroTabRegressor(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(
            :actA => :identity, :init_scale => 1.0, :depth => 4, :ntrees => 32, :stack_size => 1, :hidden_size => 1
        ),
        loss=:mse,
        nrounds=20,
        early_stopping_rounds=2,
        batchsize=2048,
        lr=1e-2,
    )

    # stack tree
    nobs = 1_000
    nfeats = 10
    x = rand(Float32, nfeats, nobs)
    feature_names = "var_" .* string.(1:nfeats)

    outsize = 1
    loss = NeuroTabModels.Losses.get_loss_fn(learner.loss)
    L = NeuroTabModels.Losses.get_loss_type(learner.loss)
    chain = learner.arch(; ins=nfeats, outsize)
    info = Dict(:nrounds => 0, :feature_names => feature_names)
    m = NeuroTabModel(L, chain, info)
end

@testset "Regression - NeuroTree" begin
    Random.seed!(123)
    X = randn(Float32, 1000, 10)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, 1000)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    target_name = "y"
    feature_names = setdiff(names(df), [target_name])

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    learner = NeuroTabRegressor(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(:depth => 3),
        loss=:mse,
        nrounds=20,
        early_stopping_rounds=2,
        lr=1e-1,
    )

    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names)

    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, deval, print_every_n=5)

    p = m(deval)
    @test size(p, 1) == nrow(deval)
    @test !any(isnan, p)
    mse_model = mean((p .- deval.y) .^ 2)
    mse_baseline = mean((mean(dtrain.y) .- deval.y) .^ 2)
    @test mse_model < mse_baseline
end

@testset "Regression - TabM $arch_type" for arch_type in [:tabm, :tabm_mini, :tabm_packed]
    Random.seed!(123)
    X = randn(Float32, 1000, 10)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, 1000)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    target_name = "y"
    feature_names = setdiff(names(df), [target_name])

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    arch = NeuroTabModels.TabMConfig(; k=4, n_blocks=2, d_block=32, dropout=0.0, arch_type)
    learner = NeuroTabRegressor(arch; loss=:mse, nrounds=20, early_stopping_rounds=2, lr=1e-2)

    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names)

    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, deval, print_every_n=5)

    p = m(deval)
    @test size(p, 1) == nrow(deval)
    @test !any(isnan, p)
    mse_model = mean((p .- deval.y) .^ 2)
    mse_baseline = mean((mean(dtrain.y) .- deval.y) .^ 2)
    @test mse_model < mse_baseline
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
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

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
    @test mean(ptrain .== levelcode.(dtrain.class)) > 0.95
    @test mean(peval .== levelcode.(deval.class)) > 0.95
end

@testset "Classification - TabM $arch_type" for arch_type in [:tabm, :tabm_mini, :tabm_packed]
    Random.seed!(123)
    X, y = @load_crabs
    df = DataFrame(X)
    df[!, :class] = y
    target_name = "class"
    feature_names = setdiff(names(df), [target_name])
    transform!(df, feature_names .=> (x -> (x .- mean(x)) ./ std(x)); renamecols=false)

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    arch = NeuroTabModels.TabMConfig(; k=4, n_blocks=1, d_block=64, dropout=0.1, arch_type)
    learner = NeuroTabClassifier(
        arch;
        embedding_config=Dict(:embedding_type => :batchnorm),
        nrounds=200,
        batchsize=32,
        early_stopping_rounds=10,
        lr=1e-2,
    )

    m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names, print_every_n=5);

    ptrain = [argmax(x) for x in eachrow(m(dtrain))]
    peval = [argmax(x) for x in eachrow(m(deval))]
    @test mean(ptrain .== levelcode.(dtrain.class)) > 0.95
    @test mean(peval .== levelcode.(deval.class)) > 0.95
end

@testset "Classification - $arch_name" for (arch_name, arch) in [
    ("MLP", NeuroTabModels.MLPConfig(; hidden_size=32, stack_size=1, dropout=0.5)),
    ("MLPAttn", NeuroTabModels.MLPAttnConfig(; hidden_size=32, nheads=4, stack_size=1, dropout=0.1)),
    (
        "NeuroTreeAttn",
        NeuroTabModels.NeuroTreeAttnConfig(; hidden_size=32, nheads=4, stack_size=1, depth=3, ntrees=8, dropout=0.1),
    ),
    ("ResNet", NeuroTabModels.ResNetConfig(; hidden_size=32, stack_size=1, dropout=0.5)),
]
    Random.seed!(123)
    X, y = @load_crabs
    df = DataFrame(X)
    df[!, :class] = y
    target_name = "class"
    feature_names = setdiff(names(df), [target_name])
    transform!(df, feature_names .=> (x -> (x .- mean(x)) ./ std(x)); renamecols=false)

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    learner = NeuroTabClassifier(
        arch;
        embedding_config=Dict(:embedding_type => :batchnorm),
        nrounds=200,
        batchsize=32,
        early_stopping_rounds=5,
        lr=2e-3,
    )

    m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names, print_every_n=5)

    ptrain = [argmax(x) for x in eachrow(m(dtrain))]
    peval = [argmax(x) for x in eachrow(m(deval))]
    @test mean(ptrain .== levelcode.(dtrain.class)) > 0.95
    @test mean(peval .== levelcode.(deval.class)) > 0.95
end

@testset "Regression - MLPAttn grouped" begin
    Random.seed!(123)
    nobs = 400
    X = randn(Float32, nobs, 8)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, nobs)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    df[!, :grp] = repeat(1:20, inner=20)
    target_name = "y"
    feature_names = setdiff(names(df), [target_name, "grp"])

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]
    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]
    sort!(dtrain, :grp)
    sort!(deval, :grp)

    arch = NeuroTabModels.MLPAttnConfig(; hidden_size=32, nheads=4, stack_size=1, n_attn_layers=1)
    learner = NeuroTabRegressor(arch; loss=:mse, nrounds=20, early_stopping_rounds=5, lr=1e-2, batchsize=64)

    m = NeuroTabModels.fit(
        learner, dtrain; target_name, feature_names, deval, group_key="grp", print_every_n=5
    )

    p = m(deval)
    @test size(p, 1) == nrow(deval)
    @test !any(isnan, p)
end

@testset "MLPAttn key-padding mask" begin
    Random.seed!(123)
    rng = Random.Xoshiro(123)
    nfeats, hsize, nheads = 6, 16, 4
    arch = NeuroTabModels.MLPAttnConfig(; hidden_size=hsize, nheads, stack_size=1, dropout=0.0)
    chain = arch(; ins=nfeats, outsize=1)
    ps, st = Lux.setup(rng, chain)
    st = Lux.testmode(st)

    x_real = randn(Float32, nfeats, 3)
    y1, _ = chain(x_real, ps, st)

    x_pad = hcat(x_real, zeros(Float32, nfeats, 2))
    w = reshape(Float32[1, 1, 1, 0, 0], 1, 1, 5)
    y2, _ = chain((x_pad, w), ps, st)

    @test size(y1, 2) == 3
    @test size(y2, 2) == 5
    @test y2[:, 1:3] ≈ y1
end

@testset "Regression - NeuroTreeAttn grouped" begin
    Random.seed!(123)
    nobs = 400
    X = randn(Float32, nobs, 8)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, nobs)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    df[!, :grp] = repeat(1:20, inner=20)
    target_name = "y"
    feature_names = setdiff(names(df), [target_name, "grp"])

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]
    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]
    sort!(dtrain, :grp)
    sort!(deval, :grp)

    arch = NeuroTabModels.NeuroTreeAttnConfig(;
        hidden_size=32, nheads=4, stack_size=1, n_attn_layers=1, depth=3, ntrees=8
    )
    learner = NeuroTabRegressor(arch; loss=:mse, nrounds=20, early_stopping_rounds=5, lr=1e-2, batchsize=64)

    m = NeuroTabModels.fit(
        learner, dtrain; target_name, feature_names, deval, group_key="grp", print_every_n=5
    )

    p = m(deval)
    @test size(p, 1) == nrow(deval)
    @test !any(isnan, p)
end

@testset "NeuroTreeAttn key-padding mask" begin
    Random.seed!(123)
    rng = Random.Xoshiro(123)
    nfeats, hsize, nheads = 6, 16, 4
    arch = NeuroTabModels.NeuroTreeAttnConfig(;
        hidden_size=hsize, nheads, stack_size=1, dropout=0.0, depth=3, ntrees=4
    )
    chain = arch(; ins=nfeats, outsize=1)
    ps, st = Lux.setup(rng, chain)
    st = Lux.testmode(st)

    x_real = randn(Float32, nfeats, 3)
    y1, _ = chain(x_real, ps, st)

    x_pad = hcat(x_real, zeros(Float32, nfeats, 2))
    w = reshape(Float32[1, 1, 1, 0, 0], 1, 1, 5)
    y2, _ = chain((x_pad, w), ps, st)

    @test size(y1, 2) == 3
    @test size(y2, 2) == 5
    @test y2[:, 1:3] ≈ y1
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
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

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

@testset "Backend/device - Classification ($backend, $device)" for (backend, device) in
                                                                   [(:enzyme, :cpu), (:zygote, :cpu)]
    Random.seed!(123)
    X, y = @load_crabs
    df = DataFrame(X)
    df[!, :class] = y
    target_name = "class"
    feature_names = setdiff(names(df), [target_name])
    transform!(df, feature_names .=> (x -> (x .- mean(x)) ./ std(x)); renamecols=false)

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    learner = NeuroTabClassifier(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(:depth => 4),
        embedding_config=Dict(:embedding_type => :batchnorm),
        nrounds=200,
        early_stopping_rounds=5,
        lr=3e-2,
        backend,
        device,
    )

    m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names)

    ptrain = [argmax(x) for x in eachrow(m(dtrain))]
    peval = [argmax(x) for x in eachrow(m(deval))]
    @test mean(ptrain .== levelcode.(dtrain.class)) > 0.95
    @test mean(peval .== levelcode.(deval.class)) > 0.95
end