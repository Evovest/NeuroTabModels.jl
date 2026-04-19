@testset "Core - data iterators" begin end

@testset "Core - internals test" begin

    learner = NeuroTabRegressor(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(
            :actA => :identity,
            :init_scale => 1.0,
            :depth => 4,
            :ntrees => 32,
            :stack_size => 1,
            :hidden_size => 1),
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
    chain = learner.arch(; nfeats, outsize)
    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
    )
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

    m = NeuroTabModels.fit(
        learner,
        dtrain;
        target_name,
        feature_names
    )

    m = NeuroTabModels.fit(
        learner,
        dtrain;
        target_name,
        feature_names,
        deval,
        print_every_n=5
    )

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

    m = NeuroTabModels.fit(
        learner,
        dtrain;
        target_name,
        feature_names
    )

    m = NeuroTabModels.fit(
        learner,
        dtrain;
        target_name,
        feature_names,
        deval,
        print_every_n=5
    )

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

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    learner = NeuroTabClassifier(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(
            :depth => 4),
        embedding_config=Dict(:embedding_type => :batchnorm),
        nrounds=200,
        early_stopping_rounds=5,
        lr=3e-2,
    )

    m = NeuroTabModels.fit(
        learner,
        dtrain;
        deval,
        target_name,
        feature_names,
    )
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

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    arch = NeuroTabModels.TabMConfig(; k=4, n_blocks=2, d_block=16, dropout=0.0, arch_type)
    learner = NeuroTabClassifier(arch;
        embedding_config=Dict(:embedding_type => :batchnorm),
        nrounds=200,
        batchsize=32,
        early_stopping_rounds=5,
        lr=1e-2,
    )

    m = NeuroTabModels.fit(
        learner,
        dtrain;
        deval,
        target_name,
        feature_names,
        print_every_n=5
    )

    ptrain = [argmax(x) for x in eachrow(m(dtrain))]
    peval = [argmax(x) for x in eachrow(m(deval))]
    @test mean(ptrain .== levelcode.(dtrain.class)) > 0.95
    @test mean(peval .== levelcode.(deval.class)) > 0.95

end