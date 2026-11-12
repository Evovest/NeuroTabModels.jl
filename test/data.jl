@testset "Core - data iterators" begin
    df = DataFrame(x=rand(Float32, 8), y=rand(Float32, 8), off=rand(Float32, 8))
    for name in ("off", :off)
        dl = NeuroTabModels.Data.get_df_loader_train(
            df; feature_names=["x"], target_name=:y, offset_name=name, batchsize=8, shuffle=false
        )
        x, y, w, offset = first(dl)
        @test w == ones(Float32, 8)
        @test vec(offset) ≈ df.off
    end
end

@testset "Regression - grouped loader honours weights" begin
    # real rows carry their weights, and the pad of the shorter group keeps zero
    df = DataFrame(x1=Float32[1, 2, 3, 4, 5, 6, 7], y=Float32[1, 2, 3, 4, 5, 6, 7],
        w=Float32[0.5, 2, 1, 3, 1, 1, 4], grp=[1, 1, 1, 2, 2, 2, 2])
    dfg = groupby(df, :grp; sort=true)
    loader = NeuroTabModels.Data.get_df_loader_train(
        dfg; feature_names=[:x1], target_name=:y, weight_name=:w, batchsize=0, shuffle=false
    )
    ws = [vec(w) for (_, _, w) in loader]
    @test ws[1] == Float32[0.5, 2, 1, 0]
    @test ws[2] == Float32[3, 1, 1, 4]
    bad = copy(df)
    bad.w = -df.w
    @test_throws "non-negative and finite" NeuroTabModels.Data.get_df_loader_train(
        groupby(bad, :grp); feature_names=[:x1], target_name=:y, weight_name=:w, batchsize=0
    )
    # a zero weight only drops the row from the loss; the mask still marks it as real
    zw = copy(df)
    zw.w = [0; df.w[2:end]]
    (_, mask), _, w = first(NeuroTabModels.Data.get_df_loader_train(
        groupby(zw, :grp; sort=true); feature_names=[:x1], target_name=:y, weight_name=:w, batchsize=0, shuffle=false
    ))
    @test mask == [true, true, true, false]
    @test vec(w) == Float32[0, 2, 1, 0]
end

@testset "Regression - grouped loader honours offsets" begin
    # real rows carry their offsets, and the pad of the shorter group keeps zero
    df = DataFrame(x1=Float32[1, 2, 3, 4, 5, 6, 7], y=Float32[1, 2, 3, 4, 5, 6, 7],
        off=Float32[0.5, -1, 2, 3, 1, -2, 4], grp=[1, 1, 1, 2, 2, 2, 2])
    dfg = groupby(df, :grp; sort=true)
    for name in ("off", :off)
        loader = NeuroTabModels.Data.get_df_loader_train(
            dfg; feature_names=[:x1], target_name=:y, offset_name=name, batchsize=0, shuffle=false
        )
        offs = [o for (_, _, _, o) in loader]
        @test size(offs[1]) == (1, 1, 4)
        @test vec(offs[1]) == Float32[0.5, -1, 2, 0]
        @test vec(offs[2]) == Float32[3, 1, -2, 4]
    end
    # several offset columns give one row each
    _, _, _, o2 = first(NeuroTabModels.Data.get_df_loader_train(
        dfg; feature_names=[:x1], target_name=:y, offset_name=[:off, :x1], batchsize=0, shuffle=false
    ))
    @test o2[:, 1, :] == Float32[0.5 -1 2 0; 1 2 3 0]

    # the metrics add each row's own offset, as a vector, a (K, B) matrix or a grouped (K, 1, B)
    M = NeuroTabModels.Metrics
    p = Float32[0.3 -1.2 0.8 2.0 -0.4; 1.1 0.2 -0.7 0.5 0.9]
    o = Float32[0.5 -1 2 0 3; 1 0 -0.5 2 1]
    y = Float32[1 0 2 3 -1]
    c = UInt32[1 2 2 1 2]
    w = Float32[1, 2, 1, 1, 3]
    for off in (o[1, :], reshape(o[1, :], 1, 1, :))
        @test M.pearson(_ -> p[1:1, :], p, y, w, off) ≈ M.pearson(_ -> p[1:1, :] .+ o[1:1, :], p, y, w)
    end
    for off in (o, reshape(o, 2, 1, :))
        @test M.gaussian_mle(_ -> p, p, y, w, off) ≈ M.gaussian_mle(_ -> p .+ o, p, y, w)
        @test M.mlogloss(_ -> p, p, c, w, off) ≈ M.mlogloss(_ -> p .+ o, p, c, w)
    end

end

