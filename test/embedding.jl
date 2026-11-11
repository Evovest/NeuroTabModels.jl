using NeuroTabModels.Models.Embeddings: EmbeddingLayer, LayerNormEmbeddings, BatchNormEmbeddings
using NeuroTabModels.Models.Embeddings: LinearEmbeddings, TemporalEmbeddings, CategoricalEmbeddings
using NeuroTabModels.Models.Embeddings: build_embedding_chain, embedding_width
using NNlib
using Statistics: var

@testset "GroupedDense" begin
    rng = Xoshiro(1)
    in_dims, out_dims, n_groups, batch = 4, 3, 5, 8
    layer = NeuroTabModels.GroupedDense(in_dims => out_dims, n_groups)
    ps, st = Lux.setup(rng, layer)
    x = randn(Float32, in_dims, n_groups, batch)
    y, st_out = layer(x, ps, st)

    @test size(y) == (out_dims, n_groups, batch)
    @test size(ps.bias) == (out_dims, n_groups, 1)
    @test st_out === st
    @test LuxCore.parameterlength(layer) == out_dims * in_dims * n_groups + out_dims * n_groups

    y_ref = Array{Float32}(undef, out_dims, n_groups, batch)
    for g in 1:n_groups
        y_ref[:, g, :] = ps.weight[:, :, g] * x[:, g, :] .+ ps.bias[:, g, 1]
    end
    @test y ≈ y_ref

    layer_nb = NeuroTabModels.GroupedDense(in_dims => out_dims, n_groups, NNlib.relu; use_bias=false)
    ps_nb, st_nb = Lux.setup(rng, layer_nb)
    y_nb, _ = layer_nb(x, ps_nb, st_nb)
    y_relu = Array{Float32}(undef, out_dims, n_groups, batch)
    for g in 1:n_groups
        y_relu[:, g, :] = NNlib.relu.(ps_nb.weight[:, :, g] * x[:, g, :])
    end
    @test y_nb ≈ y_relu
    @test !haskey(ps_nb, :bias)
end

@testset "Norm embeddings" begin
    nfeats, batch = 6, 16
    x = randn(Float32, nfeats, batch)
    rng = Xoshiro(123)
    for (etype, T) in [(:batchnorm, BatchNormEmbeddings), (:layernorm, LayerNormEmbeddings)]
        config = EmbeddingLayer(Dict(:embedding_type => etype))
        @test config.num isa T
        layer = build_embedding_chain(config, [Symbol(:x, j) for j in 1:nfeats])
        ps, st = Lux.setup(rng, layer)
        y, _ = layer(x, ps, st)
        @test size(y) == (nfeats, batch)
        @test !any(isnan, y)
        @test embedding_width(layer, x, Xoshiro(1)) == nfeats
    end
end

@testset "Embeddings - Regression" begin
    Random.seed!(123)
    n = 1000
    X = randn(Float32, n, 10)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, n)

    df = DataFrame(X, :auto)
    df[!, :y] = y
    target_name = "y"
    feature_names = setdiff(names(df), [target_name])

    train_indices = 1:800
    dtrain = df[train_indices, :]
    deval = df[801:end, :]

    mse_baseline = mean((mean(dtrain.y) .- deval.y) .^ 2)

    architectures = [
        ("TabM", NeuroTabModels.TabMConfig(; k=4, n_blocks=2, d_block=32, dropout=0.0)),
        ("NeuroTree", NeuroTabModels.NeuroTreeConfig(; depth=3)),
    ]

    @testset "$arch_name - $embedding_type" for (arch_name, arch) in architectures,
        embedding_type in [:periodic, :linear, :piecewise]

        embedding_config = Dict(:embedding_type => embedding_type, :d_embedding => 8)
        if embedding_type == :piecewise
            embedding_config[:bins] = 16
        elseif embedding_type == :periodic
            embedding_config[:frequencies] = 8
        end

        learner = NeuroTabRegressor(arch; loss=:mse, nrounds=20, lr=1e-2, embedding_config)

        m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names)

        p = m(deval)
        @test size(p, 1) == nrow(deval)
        @test !any(isnan, p)
        @test mean((p .- deval.y) .^ 2) < mse_baseline
    end
end

@testset "Embeddings - Classification" begin
    Random.seed!(123)
    X, y = @load_crabs
    df = DataFrame(X)
    df[!, :class] = y
    target_name = "class"
    feature_names = setdiff(names(df), [target_name])
    transform!(df, feature_names .=> (x -> (x .- mean(x)) ./ std(x)); renamecols=false)

    train_indices = randperm(nrow(df))[1:Int(0.8 * nrow(df))]
    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    @testset "$embedding_type" for embedding_type in [:periodic, :linear, :piecewise]
        embedding_config = Dict(:embedding_type => embedding_type, :d_embedding => 8)
        if embedding_type == :piecewise
            embedding_config[:bins] = 16
        elseif embedding_type == :periodic
            embedding_config[:frequencies] = 16
        end

        arch = NeuroTabModels.TabMConfig(; k=4, n_blocks=1, d_block=128, dropout=0.0)
        learner = NeuroTabClassifier(
            arch; nrounds=500, batchsize=64, lr=1e-2, early_stopping_rounds=20, embedding_config
        )

        m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names, print_every_n=5);

        ptrain = [argmax(x) for x in eachrow(m(dtrain))]
        peval = [argmax(x) for x in eachrow(m(deval))]
        @test mean(ptrain .== levelcode.(dtrain.class)) >= 0.95
        @test mean(peval .== levelcode.(deval.class)) >= 0.95
    end
end

@testset "Temporal and categorical embeddings by name" begin
    Random.seed!(123)
    n = 2000
    effect = Dict("a" => -2.0, "b" => 0.0, "c" => 1.0, "d" => 3.0)
    store = rand(collect(keys(effect)), n)
    x1, t = randn(n), Float64.(1:n)
    y = x1 .+ getindex.(Ref(effect), store) .+ sin.(2π .* t ./ 50) .+ 0.1 .* randn(n)
    df = DataFrame(; x1, t, store=categorical(store), y)
    dtrain, deval = df[1:1600, :], df[1601:end, :]

    embedding_config = EmbeddingLayer(;
        num=LinearEmbeddings(; d_embedding=8),
        temp=TemporalEmbeddings(; feature=:t, order=[3], periods=[50.0], d_embedding=8),
        cat=CategoricalEmbeddings(; features=[:store], d_embedding=4),
    )
    learner = NeuroTabRegressor(NeuroTabModels.MLPConfig(); nrounds=60, lr=3e-3, batchsize=128, embedding_config)
    m = NeuroTabModels.fit(learner, dtrain; feature_names=[:x1, :t, :store], target_name=:y)

    p = m(deval)
    @test mean((p .- deval.y) .^ 2) < 0.1 * var(deval.y)
    @test m.info[:cat_levels] == Dict(:store => ["a", "b", "c", "d"])

    dnew = deval[1:3, :]
    dnew.store = categorical(["unseen", "a", "d"])
    @test !any(isnan, m(dnew))

    config = EmbeddingLayer(Dict(  # as parsed from JSON
        "temporal" => Dict("feature" => "t", "order" => Any[3], "periods" => Any[50]),
        "categorical" => Dict("features" => Any["store"]),
    ))
    @test config.temp.feature == :t
    @test config.temp.periods == Float32[50]
    @test config.cat.features == [:store]
    config = EmbeddingLayer(; num=:linear, temp=:t, cat=[:store])
    @test config.num isa LinearEmbeddings
    @test config.temp.feature == :t
    @test config.cat.features == [:store]
    @test_throws ArgumentError EmbeddingLayer(; num=:foo)

    learner = NeuroTabRegressor(
        NeuroTabModels.MLPConfig(); nrounds=1, embedding_config=EmbeddingLayer(; num=:linear, temp=:t)
    )
    m = NeuroTabModels.fit(learner, dtrain; feature_names=[:x1, :t, :store], target_name=:y)
    @test m.info[:cat_levels] == Dict(:store => ["a", "b", "c", "d"])
    m = NeuroTabModels.fit(learner, dtrain; feature_names=[:x1, :t], target_name=:y)
    delete!(m.info, :cat_levels)
    @test !any(isnan, m(deval))
    m = NeuroTabModels.fit(
        NeuroTabRegressor(NeuroTabModels.MLPConfig(); nrounds=1), dtrain; feature_names=[:store], target_name=:y
    )
    @test !any(isnan, m(deval))

    @test_throws "also listed as categorical" EmbeddingLayer(;
        temp=TemporalEmbeddings(; feature=:t), cat=CategoricalEmbeddings(; features=[:t])
    )
    @test_throws "not in feature_names" NeuroTabModels.fit(
        NeuroTabRegressor(NeuroTabModels.MLPConfig(); nrounds=1, embedding_config=EmbeddingLayer(; temp=TemporalEmbeddings(; feature=:date))),
        dtrain; feature_names=[:x1, :t], target_name=:y,
    )
end
