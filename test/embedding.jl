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

    @testset "$arch_name - $et" for (arch_name, arch) in architectures,
                                     et in [:periodic, :linear, :piecewise]

        embed_kw = Dict(:embedding_type => et, :d_embedding => 8)
        if et == :piecewise
            embed_kw[:n_bins] = 16
        end

        learner = NeuroTabRegressor(arch;
            loss=:mse, nrounds=20, lr=1e-2,
            embedding_config=embed_kw)

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

    train_indices = randperm(nrow(df))[1:Int(0.8 * nrow(df))]
    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    @testset "$et" for et in [:periodic, :linear, :piecewise]
    
    
        embed_kw = Dict(:embedding_type => et, :d_embedding => 8)
        if et == :piecewise
            embed_kw[:n_bins] = 16
        end
        if et == :periodic
            embed_kw[:n_frequencies] = 8
        end

        arch = NeuroTabModels.TabMConfig(; k=2, n_blocks=2, d_block=16, dropout=0.0)
        learner = NeuroTabClassifier(arch;
            nrounds=1000, batchsize=32, early_stopping_rounds=50, lr=1e-2,
            embedding_config=embed_kw)

        m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names)

        ptrain = [argmax(x) for x in eachrow(m(dtrain))]
        peval = [argmax(x) for x in eachrow(m(deval))]
        @test mean(ptrain .== levelcode.(dtrain.class)) > 0.95
        @test mean(peval .== levelcode.(deval.class)) > 0.95

    end

end