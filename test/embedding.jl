@testset "Numerical Embeddings" begin

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

    # (name, base config kwargs)
    architectures = [
        ("TabM", Dict(
            :k => 4, :n_blocks => 2, :d_block => 32, :dropout => 0.0)),
    ]

    @testset "$arch_name - $et" for (arch_name, base_kw) in architectures,
                                     et in [:periodic, :linear, :piecewise]

        extra = et == :piecewise ? Dict(:n_bins => 16) : Dict()
        arch = NeuroTabModels.TabMConfig(;
            base_kw..., use_embeddings=true, d_embedding=8, embedding_type=et, extra...)

        learner = NeuroTabRegressor(arch; loss=:mse, nrounds=20, lr=1e-2)
        m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names)

        p = m(deval)
        @test size(p, 1) == nrow(deval)
        @test !any(isnan, p)
        @test mean((p .- deval.y) .^ 2) < mse_baseline

    end

end