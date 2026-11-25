@testset "GaussianMLE inverse link and scaling" begin
    pred = Float32[0.2 0.0; 0.3 1.0]
    p = NeuroTabModels.Infer._inverse_link(NeuroTabModels.Losses.GaussianMLE(), pred)
    @test size(p) == (2, 2)
    @test p[:, 1] ≈ Float32[0.2, 0.3]
    @test p[:, 2] ≈ exp.(Float32[0.0, 1.0])

    scalers = (mu=10.0f0, sigma=2.0f0)
    p_scaled = NeuroTabModels.Infer._scaler(NeuroTabModels.Losses.GaussianMLE(), copy(p), scalers)
    @test p_scaled[:, 1] ≈ p[:, 1] .* 2 .+ 10
    @test p_scaled[:, 2] ≈ p[:, 2] .* 2
end

@testset "Target input checks" begin
    Random.seed!(123)
    n = 200
    df = DataFrame(randn(Float32, n, 3), :auto)
    df.y = df.x1 .+ 0.1f0 .* randn(Float32, n)
    learner = NeuroTabRegressor(NeuroTabModels.TabMConfig(; k=2, d_block=16, n_blocks=1); nrounds=1)
    fit1(d; feature_names=["x1", "x2", "x3"]) = NeuroTabModels.fit(learner, d; target_name="y", feature_names)

    @test_throws "feature_names" fit1(df; feature_names=["x1", "y"])
    dflat = copy(df); dflat.y .= 1
    @test_throws "constant" fit1(dflat)
    dnan = copy(df); dnan.y[5] = NaN32
    @test_throws "missing or NaN" fit1(dnan)

    # The metric must read the outputs the loss gives: two per target for `:gaussian_mle`, one otherwise.
    # Only an eval set reads the metric, so only a fit with one checks it.
    fitlm(loss, metric; deval=df) = NeuroTabModels.fit(
        NeuroTabRegressor(NeuroTabModels.MLPConfig(); loss, metric, nrounds=1), df;
        target_name="y", feature_names=["x1", "x2", "x3"], deval,
    )
    @test_throws "`metric=:mse` does not fit `loss=:gaussian_mle`" fitlm(:gaussian_mle, :mse)
    @test_throws "`metric=:gaussian_mle` does not fit `loss=:mse`" fitlm(:mse, :gaussian_mle)
    @test fitlm(:gaussian_mle, :pearson) isa NeuroTabModels.NeuroTabModel
    @test fitlm(:gaussian_mle, :mse; deval=nothing) isa NeuroTabModels.NeuroTabModel
end

@testset "Multiple targets" begin
    Random.seed!(123)
    n = 500
    X = randn(Float32, n, 4)
    df = DataFrame(X, :auto)
    df.y1, df.y2, df.y3 = X[:, 1], X[:, 2] .- X[:, 3], sin.(X[:, 4])
    df.y1k = 1000 .* df.y1
    df.w = rand(Float32, n) .+ 0.5f0
    feature_names = ["x1", "x2", "x3", "x4"]
    arch = NeuroTabModels.TabMConfig(; k=4, d_block=32, n_blocks=1, dropout=0.0)
    fit_mt(loss, target_name; kw...) = NeuroTabModels.fit(
        NeuroTabRegressor(arch; loss, nrounds=3, lr=1e-2, batchsize=128), df;
        feature_names, target_name, deval=df, kw...,
    )

    @test size(fit_mt(:mse, ["y1", "y2", "y3"]; weight_name="w")(df)) == (n, 3)
    # Each target is unscaled with its own mean and std.
    p = fit_mt(:mse, ["y1", "y1k"])(df)
    @test 500 < std(p[:, 2]) / std(p[:, 1]) < 2000

    # Gaussian outputs interleave per target: μ₁, σ₁, μ₂, σ₂.
    μ, σ = reshape(Float32[1, 2, 10, 20], 2, 1, 2), reshape(Float32[3, 4, 30, 40], 2, 1, 2)
    @test vec(NeuroTabModels.Models.interleave_heads(μ, σ)) == Float32[1, 3, 2, 4, 10, 30, 20, 40]
    NI, G = NeuroTabModels.Infer, NeuroTabModels.Losses.GaussianMLE()
    @test NI._scaler(G, NI._inverse_link(G, Float32[1 0 2 log(2)]), (mu=Float32[10, 20], sigma=Float32[2, 3])) ≈
          Float32[12 2 26 6]
    @test size(fit_mt(:gaussian_mle, ["y1", "y2"])(df)) == (n, 4)

    # Each target is paired with its own offset: predictions leave the offset out and stay near 1.
    df.o1, df.o2 = fill(1.5f0, n), fill(-1.5f0, n)
    df.e1, df.e2 = exp.(df.o1), exp.(df.o2)
    p = NeuroTabModels.fit(
        NeuroTabRegressor(arch; loss=:tweedie, nrounds=5, lr=1e-2, batchsize=128), df;
        feature_names, target_name=["e1", "e2"], offset_name=["o1", "o2"],
    )(df)
    @test all(x -> 0.8 < x < 1.25, p)

    # Targets are averaged, not summed, in the loss and the eval steps.
    L, MT, CB = NeuroTabModels.Losses, NeuroTabModels.Metrics, NeuroTabModels.Fit.CallBacks
    idm = (x, ps, st) -> (x, st)
    pred, y, w = randn(Float32, 2, 1, 8), randn(Float32, 2, 1, 8), ones(Float32, 8)
    @test first(L.MSE()(idm, nothing, nothing, (pred, y))) ≈ first(L.MSE()(idm, nothing, nothing, (pred, y, w)))
    p2, y2 = dropdims(pred; dims=2), dropdims(y; dims=2)
    for d in ((pred, y2), (pred, y2, w))
        num, den = CB._build_eval_step(idm, MT.mse, d, nothing, nothing; reactant=false)(d..., nothing, nothing)
        @test num / den ≈ mean((p2 .- y2) .^ 2)
    end

    # Pearson takes the mean over targets; with two outputs per target the metric reads the μ rows.
    pr, yr, wr = randn(Float32, 2, 1, 16), randn(Float32, 2, 16), rand(Float32, 16) .+ 0.5f0
    ploss(t) = first(L.Pearson()(idm, nothing, nothing, (pr[t:t, :, :], yr[t:t, :], wr)))
    @test first(L.Pearson()(idm, nothing, nothing, (pr, yr, wr))) ≈ (ploss(1) + ploss(2)) / 2
    pg = randn(Float32, 4, 16)
    @test MT.pearson(_ -> pg, pg, yr, wr) ≈ MT.pearson(_ -> pg[[1, 3], :], pg, yr, wr)

    # Grouped targets are `(T, 1, bs)`, zero on pads.
    dg = DataFrame(x1=Float32[1, 2, 3, 4, 5], y1=Float32[1, 2, 3, 4, 5], y2=Float32[5, 4, 3, 2, 1], g=[1, 1, 1, 2, 2])
    _, yg, _ = first(NeuroTabModels.Data.get_df_loader_train(
        groupby(dg, :g; sort=true); feature_names=[:x1], target_name=[:y1, :y2], batchsize=0, shuffle=false
    ))
    @test yg == reshape(Float32[1 2 3; 5 4 3], 2, 1, 3)

    df.cls = categorical(rand(["a", "b"], n))
    clf = NeuroTabClassifier(arch; nrounds=1)
    # A one-name vector is a single target, in the eval set too.
    @test size(NeuroTabModels.fit(clf, df; feature_names, target_name=["cls"], deval=df)(df)) == (n, 2)
    @test_throws "Multiple targets" NeuroTabModels.fit(clf, df; feature_names, target_name=["cls", "y1"])
    @test_throws "`cls` must be numeric" fit_mt(:mse, ["y1", "cls"])
    @test_throws "2 columns, expected 1 or 4" fit_mt(:gaussian_mle, ["y1", "y2"]; offset_name=["o1", "o2"])
    @test_throws "duplicate" fit_mt(:mse, ["y1", "y1"])
    @test_throws "`x1` is also listed" fit_mt(:mse, ["y1", "x1"])
    split = NeuroTabRegressor(NeuroTabModels.MLPConfig(; MLE_tree_split=true); nrounds=1)
    @test_throws "requires `loss=:gaussian_mle`" NeuroTabModels.fit(split, df; feature_names, target_name=["y1", "y2"])
end

@testset "Poisson loss and metric" begin
    L = NeuroTabModels.Losses
    M = NeuroTabModels.Metrics
    idm = (x, ps, st) -> (x, st)

    @test L.LossType(:poisson) === L.Poisson()
    @test !M.is_maximise(M.poisson)

    y = Float32[0, 1, 3, 7]
    η = Float32[0.2, -0.5, 1.0, 2.0]
    μ = exp.(Float64.(η))
    yd = Float64.(y)
    dev = 2 .* (ifelse.(yd .> 0, yd .* log.(yd), 0.0) .- yd .* log.(μ) .+ μ .- yd)

    val, _, _ = L.Poisson()(idm, (;), (;), (reshape(η, 1, 1, :), y))
    @test val ≈ mean(dev) rtol = 1e-5
    @test M.poisson(identity, reshape(η, 1, :), y; agg=sum) ≈ sum(dev) rtol = 1e-5

    w = Float32[1, 2, 1, 0.5]
    valw, _, _ = L.Poisson()(idm, (;), (;), (reshape(η, 1, 1, :), y, w))
    @test valw ≈ sum(dev .* w) / sum(w) rtol = 1e-5

    offset = Float32[0.1, 0.1, 0.1, 0.1]
    @test M.poisson(identity, reshape(η, 1, :), y, w, offset; agg=sum) ≈
          M.poisson(identity, reshape(η .+ offset, 1, :), y, w; agg=sum)
end

@testset "Pearson loss and metric" begin
    L = NeuroTabModels.Losses
    M = NeuroTabModels.Metrics
    idm = (x, ps, st) -> (x, st)
    pred_fn = x -> x

    # The loss takes a model's `(outsize, k, batch)` output, the metric its `(outsize, batch)` mean over k.
    x = Float32[1.0 2.0 3.0 4.0]
    y = Float32[1.0 2.0 3.0 4.0]
    x3 = reshape(x, 1, 1, 4)
    val, _, _ = L.Pearson()(idm, (;), (;), (x3, y))
    @test val ≈ -1.0f0
    @test M.pearson(pred_fn, x, y) ≈ 4.0f0
    @test M.is_maximise(M.pearson)
    @test_throws MethodError L.Pearson()(idm, (;), (;), (x, y))

    w = Float32[1, 1, 1, 1]
    valw, _, _ = L.Pearson()(idm, (;), (;), (x3, y, w))
    @test valw ≈ -1.0f0
    @test M.pearson(pred_fn, x, y, w) ≈ 4.0f0

    # Loss is -cor (mean-scale, like MSE). Metric is cor * n so get_metric can average batches.
    x_imp = Float32[1.0 2.0 3.0 4.0]
    y_imp = Float32[1.0 3.0 2.0 8.0]
    c = cor(vec(Float64.(x_imp)), vec(Float64.(y_imp)))
    val_imp, _, _ = L.Pearson()(idm, (;), (;), (reshape(x_imp, 1, 1, 4), y_imp))
    @test val_imp ≈ -c rtol = 1e-5
    @test M.pearson(pred_fn, x_imp, y_imp) ≈ c * 4 rtol = 1e-5
    @test M.pearson(pred_fn, x_imp, -y_imp) ≈ -c * 4 rtol = 1e-5

    # First output channel only; ensemble axis is averaged.
    pred2 = cat(x, x; dims=1)  # (2, 4): extra channel must be ignored
    @test M.pearson(_ -> pred2, x, y) ≈ 4.0f0
    ens = reshape(Float32[1 2 3 4; 1 2 3 4], 1, 2, 4)
    val_ens, _, _ = L.Pearson()(idm, (;), (;), (ens, y))
    @test val_ens ≈ -1.0f0

    offset = Float32[1, 1, 1, 1]
    # offset shifts p by +1; correlation with y=x is unchanged
    @test M.pearson(pred_fn, x, y, w, offset) ≈ 4.0f0

    g = Zygote.gradient(xx -> first(L.Pearson()(idm, (;), (;), (xx, y_imp))), reshape(x_imp, 1, 1, 4))
    @test g[1] isa AbstractArray
    @test all(isfinite, g[1])
end

@testset "Pearson grouped sampling" begin
    L = NeuroTabModels.Losses
    M = NeuroTabModels.Metrics
    idm = (x, ps, st) -> (reshape(x[1, :], 1, 1, size(x, 2)), st)
    pred_fn = x -> reshape(x[1, :], 1, size(x, 2))

    # Zero-weight pad must be dropped. A (0, 0) pad is nearly collinear with
    # y = x + 1, so use an off-line pad so unweighted Pearson is actually wrong.
    p_pad = Float32[1.0 2.0 3.0 0.0]
    y_pad = Float32[2.0 3.0 4.0 10.0]
    w_pad = Float32[1, 1, 1, 0]
    c_real = cor(Float64[1, 2, 3], Float64[2, 3, 4])
    c_full = cor(vec(Float64.(p_pad)), vec(Float64.(y_pad)))
    val_mask, _, _ = L.Pearson()(idm, (;), (;), (p_pad, y_pad, w_pad))
    val_full, _, _ = L.Pearson()(idm, (;), (;), (p_pad, y_pad))
    @test val_mask ≈ -c_real rtol = 1e-5
    @test val_full ≈ -c_full rtol = 1e-5
    @test M.pearson(pred_fn, p_pad, y_pad, w_pad) ≈ c_real * 3 rtol = 1e-5
    @test abs(val_full - val_mask) > 0.1
    g = Zygote.gradient(xx -> first(L.Pearson()(idm, (;), (;), (xx, y_pad, w_pad))), p_pad)
    @test g[1][4] ≈ 0 atol = 1e-6

    # Unequal groups: loader pads to max group size; eval is size-weighted mean of per-group Pearson.
    df = DataFrame(
        x1=Float32[1, 2, 3, 10, 20, 30, 40, 5, 6, 7, 8],
        y=Float32[2, 3, 4, 11, 21, 31, 41, 6, 8, 7, 20],
        grp=[1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
    )
    dfg = groupby(df, :grp; sort=true)
    loader = NeuroTabModels.Data.get_df_loader_train(
        dfg; feature_names=[:x1], target_name=:y, batchsize=0, shuffle=false
    )

    group_cors = Float64[]
    group_ns = Float64[]
    metric_acc = 0.0
    for ((x, mask), yb, w) in loader
        n = sum(mask)
        push!(group_cors, cor(Float64.(x[1, mask]), Float64.(vec(yb)[mask])))
        push!(group_ns, n)
        metric_acc += M.pearson(pred_fn, x, yb, w)
    end
    grouped_metric = metric_acc / sum(group_ns)
    @test group_ns == [3.0, 4.0, 4.0]
    @test grouped_metric ≈ sum(group_cors .* group_ns) / sum(group_ns) rtol = 1e-5
    @test abs(grouped_metric - cor(Float64.(df.x1), Float64.(df.y))) > 1e-3
end

@testset "Pearson on flat groups" begin
    L = NeuroTabModels.Losses
    M = NeuroTabModels.Metrics
    idm = (x, ps, st) -> (x, st)
    pred_fn = x -> x
    y = Float32[1.0 3.0 2.0 8.0]
    w = Float32[1, 1, 1, 1]

    # A constant prediction scores 0 instead of 0/0, and its gradient still points along y.
    p0 = fill(0.3f0, 1, 1, 4)
    val, _, _ = L.Pearson()(idm, (;), (;), (p0, y, w))
    @test val ≈ 0 atol = 1e-5
    @test M.pearson(pred_fn, p0[:, 1, :], y, w) ≈ 0 atol = 1e-5
    g = Zygote.gradient(pp -> first(L.Pearson()(idm, (;), (;), (pp, y, w))), p0)[1]
    @test cor(vec(-g), vec(y)) > 0.99

    # A flat target or a single row scores 0 as well.
    p = reshape(Float32[1.0 2.0 3.0 4.0], 1, 1, 4)
    @test M.pearson(pred_fn, p[:, 1, :], fill(2.0f0, 1, 4), w) ≈ 0 atol = 1e-5
    @test M.pearson(pred_fn, Float32[0.5;;], Float32[2.0;;], Float32[1]) ≈ 0 atol = 1e-5
end

@testset "Huber loss and metric" begin
    L = NeuroTabModels.Losses
    M = NeuroTabModels.Metrics
    idm = (x, ps, st) -> (x, st)

    @test L.LossType(:huber) === L.Huber()
    @test L.scales_target(L.Huber())
    @test L.noutputs(L.Huber()) == 1
    @test !M.is_maximise(M.huber)

    # δ = 1: quadratic inside, linear outside. Mean reduction, like MSE.
    r = Float32[0.0, 0.5, 1.0, 2.0, -3.0]
    δ = 1.0f0
    expected = ifelse.(abs.(r) .<= δ, (r .^ 2) ./ 2, δ .* (abs.(r) .- δ / 2))
    pred = reshape(r, 1, 1, :)
    y = zeros(Float32, length(r))
    val, _, _ = L.Huber()(idm, (;), (;), (pred, y))
    @test val ≈ mean(expected)
    @test M.huber(_ -> reshape(r, 1, :), r, y; agg=sum) ≈ sum(expected)

    w = Float32[1, 2, 0, 1, 0.5]
    valw, _, _ = L.Huber()(idm, (;), (;), (pred, y, w))
    @test valw ≈ sum(expected .* w) / sum(w)
    @test M.huber(_ -> reshape(r, 1, :), r, reshape(y, 1, :), w; agg=sum) ≈ sum(expected .* w)

    offset = Float32[0.25, -0.25, 0.5, 1.5, -0.5]
    valo, _, _ = L.Huber()(idm, (;), (;), (pred, y, w, offset))
    ro = r .+ offset
    expected_o = ifelse.(abs.(ro) .<= δ, (ro .^ 2) ./ 2, δ .* (abs.(ro) .- δ / 2))
    @test valo ≈ sum(expected_o .* w) / sum(w)
    @test M.huber(identity, reshape(r, 1, :), reshape(y, 1, :), w, offset; agg=sum) ≈ sum(expected_o .* w)

    # Derivative is clamp(r, -δ, δ), divided by the number of elements in the mean.
    g = Zygote.gradient(pp -> first(L.Huber()(idm, (;), (;), (pp, y))), pred)[1]
    @test vec(g) ≈ clamp.(r, -δ, δ) ./ length(r)

    # Two targets are averaged, not summed.
    pred2 = randn(Float32, 2, 1, 6)
    y2 = randn(Float32, 2, 1, 6)
    w2 = rand(Float32, 6) .+ 0.2f0
    one_target(t) = first(L.Huber()(idm, (;), (;), (pred2[t:t, :, :], y2[t:t, :, :], w2)))
    @test first(L.Huber()(idm, (;), (;), (pred2, y2, w2))) ≈ (one_target(1) + one_target(2)) / 2
end
