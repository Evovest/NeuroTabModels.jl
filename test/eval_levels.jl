@testset "Eval target coded against the training levels" begin
    rng = Xoshiro(1)
    n = 600
    df = DataFrame(randn(rng, Float32, n, 3), [:a, :b, :c])
    df.y = categorical(ifelse.(df.a .< -0.5f0, "A", ifelse.(df.a .< 0.5f0, "B", "C")))
    # eval set without class "A", built as a fresh categorical with levels B, C
    deval = df[df.y .!= "A", :]
    deval.y = categorical(String.(deval.y))
    learner = NeuroTabClassifier(; arch_name="NeuroTreeConfig", arch_config=Dict(:depth => 3), nrounds=5,
        batchsize=1024, device=:cpu)
    m = NeuroTabModels.fit(learner, df; deval, feature_names=[:a, :b, :c], target_name=:y, verbosity=0)
    p = m(deval)
    k = indexin(String.(deval.y), ["A", "B", "C"])
    @test m.info[:logger][:best_metric] ≈ -mean(log.(p[CartesianIndex.(1:nrow(deval), k)])) rtol = 1e-4

    deval.y = categorical(ifelse.(deval.y .== "B", "B", "D"))
    @test_throws ErrorException NeuroTabModels.fit(learner, df; deval, feature_names=[:a, :b, :c], target_name=:y,
        verbosity=0)
end
