@testset "Embeddings - Regression" begin

    Random.seed!(123)
    n = 1000
    X0 = randn(Float32, n, 10)
    noise = 0.1f0 .* randn(Float32, n)

    architectures = [
        ("TabM", NeuroTabModels.TabMConfig(; k=4, n_blocks=2, d_block=32, dropout=0.0)),
        ("NeuroTree", NeuroTabModels.NeuroTreeConfig(; depth=3)),
    ]

    embedding_cases = [
        (:periodic, false, () -> NeuroTabModels.EmbeddingLayer(
            num=NeuroTabModels.PeriodicEmbeddings(d_embedding=8, frequencies=8))),
        (:periodic, true, () -> NeuroTabModels.EmbeddingLayer(
            num=NeuroTabModels.PeriodicEmbeddings(d_embedding=8, frequencies=8),
            temp=NeuroTabModels.TemporalEmbeddings(index=1,
                order=Int[2, 1, 1, 0],
                periods=Float32[31_557_600, 2_629_800, 604_800, 86_400],
                trend=true, d_embedding=8))),
        (:linear, false, () -> NeuroTabModels.EmbeddingLayer(
            num=NeuroTabModels.LinearEmbeddings(d_embedding=8))),
        (:linear, true, () -> NeuroTabModels.EmbeddingLayer(
            num=NeuroTabModels.LinearEmbeddings(d_embedding=8),
            temp=NeuroTabModels.TemporalEmbeddings(index=1,
                order=Int[2, 1, 1, 0],
                periods=Float32[31_557_600, 2_629_800, 604_800, 86_400],
                trend=true, d_embedding=8))),
        (:piecewise, false, () -> NeuroTabModels.EmbeddingLayer(
            num=NeuroTabModels.PiecewiseLinearEmbeddings(d_embedding=8, bins=16))),
    ]

    @testset "$arch_name - $embedding_type$(temporal ? " temporal" : "")" for (arch_name, arch) in architectures,
        (embedding_type, temporal, build_embedding) in embedding_cases

        X = copy(X0)
        temporal && (X[:, 1] .= collect(Float32, 1:n))
        y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ noise

        df = DataFrame(X, :auto)
        df[!, :y] = y
        target_name = "y"
        feature_names = setdiff(names(df), [target_name])

        train_indices = 1:800
        dtrain = df[train_indices, :]
        deval = df[801:end, :]

        mse_baseline = mean((mean(dtrain.y) .- deval.y) .^ 2)

        learner = NeuroTabRegressor(arch;
            loss=:mse, nrounds=20, lr=1e-2,
            embedding_config=build_embedding())

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
    n = length(y)
    train_indices = randperm(n)[1:Int(0.8 * n)]

    classification_cases = [
        (:periodic, false, () -> NeuroTabModels.EmbeddingLayer(
            num=NeuroTabModels.PeriodicEmbeddings(d_embedding=8, frequencies=8))),
        (:periodic, true, () -> NeuroTabModels.EmbeddingLayer(
            num=NeuroTabModels.PeriodicEmbeddings(d_embedding=8, frequencies=8),
            temp=NeuroTabModels.TemporalEmbeddings(index=1,
                order=Int[2, 1, 1, 0],
                periods=Float32[31_557_600, 2_629_800, 604_800, 86_400],
                trend=true, d_embedding=8))),
        (:linear, false, () -> NeuroTabModels.EmbeddingLayer(
            num=NeuroTabModels.LinearEmbeddings(d_embedding=8))),
        (:linear, true, () -> NeuroTabModels.EmbeddingLayer(
            num=NeuroTabModels.LinearEmbeddings(d_embedding=8),
            temp=NeuroTabModels.TemporalEmbeddings(index=1,
                order=Int[2, 1, 1, 0],
                periods=Float32[31_557_600, 2_629_800, 604_800, 86_400],
                trend=true, d_embedding=8))),
        (:piecewise, false, () -> NeuroTabModels.EmbeddingLayer(
            num=NeuroTabModels.PiecewiseLinearEmbeddings(d_embedding=8, bins=16))),
    ]

    @testset "$embedding_type$(temporal ? " temporal" : "")" for (embedding_type, temporal, build_embedding) in classification_cases

        df = DataFrame(X)
        temporal && insertcols!(df, 1, :t => collect(Float32, 1:n))
        df[!, :class] = y
        target_name = "class"
        feature_names = setdiff(names(df), [target_name])

        dtrain = df[train_indices, :]
        deval = df[setdiff(1:n, train_indices), :]

        arch = NeuroTabModels.TabMConfig(; k=2, n_blocks=2, d_block=16, dropout=0.0, scaling_init=:random_signs)
        learner = NeuroTabClassifier(arch;
            nrounds=500, batchsize=32, early_stopping_rounds=100, lr=1e-2,
            embedding_config=build_embedding())

        m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names)

        ptrain = [argmax(x) for x in eachrow(m(dtrain))]
        peval = [argmax(x) for x in eachrow(m(deval))]
        @test mean(ptrain .== levelcode.(dtrain.class)) > 0.95
        @test mean(peval .== levelcode.(deval.class)) > 0.95

    end

end
