@testset "Eval metric scores every eval row ($backend)" for backend in (:zygote, :reactant)
    rng = Xoshiro(1)
    n = 1000
    df = DataFrame(randn(rng, Float32, n, 3), [:a, :b, :c])
    df.y = df.a .+ 0.1f0 .* randn(rng, Float32, n)
    deval = copy(df)
    # 1000 rows in batches of 512: the last 488 rows only count if the short batch is kept
    deval.y[513:end] .+= 10.0f0
    learner = NeuroTabRegressor(; arch_name="NeuroTreeConfig", arch_config=Dict(:depth => 3), loss=:mse,
        nrounds=2, batchsize=512, scale_target=false, backend, device=:cpu)
    m = NeuroTabModels.fit(learner, df; deval, feature_names=[:a, :b, :c], target_name=:y, verbosity=0)
    p = m(deval)
    @test m.info[:logger][:best_metric] ≈ mean((p .- deval.y) .^ 2) rtol = 1e-4
end
