using NeuroTabModels.Models.Embeddings: EmbeddingLayer, LayerNormEmbeddings, BatchNormEmbeddings
using NeuroTabModels.Models.Embeddings: build_embedding_chain, embedding_width
using NNlib

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
        layer = build_embedding_chain(config, nfeats)
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
        @test mean(ptrain .== levelcode.(dtrain.class)) > 0.95
        @test mean(peval .== levelcode.(deval.class)) > 0.95
    end
end