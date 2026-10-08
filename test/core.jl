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

@testset "Core - internals test" begin
    learner = NeuroTabRegressor(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(
            :actA => :identity, :init_scale => 1.0, :depth => 4, :ntrees => 32, :stack_size => 1, :hidden_size => 1
        ),
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
    loss = NeuroTabModels.Losses.LossType(learner.loss)
    chain = learner.arch(; ins=nfeats, outsize)
    info = Dict(:nrounds => 0, :feature_names => feature_names)
    m = NeuroTabModel(loss, chain, info)
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
    train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]

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

    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names)

    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, deval, print_every_n=5)

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
    train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    arch = NeuroTabModels.TabMConfig(; k=4, n_blocks=2, d_block=32, dropout=0.0, arch_type)
    learner = NeuroTabRegressor(arch; loss=:mse, nrounds=20, early_stopping_rounds=2, lr=1e-2)

    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names)

    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, deval, print_every_n=5)

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
    transform!(df, feature_names .=> (x -> (x .- mean(x)) ./ std(x)); renamecols=false)

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    learner = NeuroTabClassifier(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(:depth => 4),
        embedding_config=Dict(:embedding_type => :batchnorm),
        nrounds=200,
        early_stopping_rounds=5,
        lr=3e-2,
    )

    m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names)
    # Predictions depend on the number of samples in the dataset
    ptrain = [argmax(x) for x in eachrow(m(dtrain))]
    peval = [argmax(x) for x in eachrow(m(deval))]
    @test mean(ptrain .== levelcode.(dtrain.class)) >= 0.95
    @test mean(peval .== levelcode.(deval.class)) >= 0.95
end

@testset "Classification - TabM $arch_type" for arch_type in [:tabm, :tabm_mini, :tabm_packed]
    Random.seed!(123)
    X, y = @load_crabs
    df = DataFrame(X)
    df[!, :class] = y
    target_name = "class"
    feature_names = setdiff(names(df), [target_name])
    transform!(df, feature_names .=> (x -> (x .- mean(x)) ./ std(x)); renamecols=false)

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    arch = NeuroTabModels.TabMConfig(; k=4, n_blocks=1, d_block=64, dropout=0.1, arch_type)
    learner = NeuroTabClassifier(
        arch;
        embedding_config=Dict(:embedding_type => :batchnorm),
        nrounds=200,
        batchsize=32,
        early_stopping_rounds=10,
        lr=1e-2,
    )

    m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names, print_every_n=5)

    ptrain = [argmax(x) for x in eachrow(m(dtrain))]
    peval = [argmax(x) for x in eachrow(m(deval))]
    @test mean(ptrain .== levelcode.(dtrain.class)) >= 0.95
    @test mean(peval .== levelcode.(deval.class)) >= 0.95
end

@testset "Classification - $arch_name" for (arch_name, arch) in [
    ("MLP", NeuroTabModels.MLPConfig(; hidden_size=32, stack_size=1, dropout=0.5)),
    ("MLPAttn", NeuroTabModels.MLPAttnConfig(; hidden_size=64, nheads=1, stack_size=1, dropout=0.2)),
    (
        "NeuroTreeAttn",
        NeuroTabModels.NeuroTreeAttnConfig(; hidden_size=8, nheads=1, depth=3, ntrees=4, dropout=0.2),
    ),
    ("ResNet", NeuroTabModels.ResNetConfig(; hidden_size=32, stack_size=1, dropout=0.2)),
]
    Random.seed!(123)
    X, y = @load_crabs
    df = DataFrame(X)
    df[!, :class] = y
    target_name = "class"
    feature_names = setdiff(names(df), [target_name])
    transform!(df, feature_names .=> (x -> (x .- mean(x)) ./ std(x)); renamecols=false)

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    learner = NeuroTabClassifier(
        arch;
        embedding_config=Dict(:embedding_type => :batchnorm),
        nrounds=200,
        batchsize=32,
        early_stopping_rounds=5,
        lr=3e-3,
    )

    m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names, print_every_n=5)

    ptrain = [argmax(x) for x in eachrow(m(dtrain))]
    peval = [argmax(x) for x in eachrow(m(deval))]
    @test mean(ptrain .== levelcode.(dtrain.class)) >= 0.95
    @test mean(peval .== levelcode.(deval.class)) >= 0.95
end

@testset "Regression - MLPAttn grouped" begin
    Random.seed!(123)
    nobs = 400
    X = randn(Float32, nobs, 8)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, nobs)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    df[!, :grp] = repeat(1:20, inner=20)
    target_name = "y"
    feature_names = setdiff(names(df), [target_name, "grp"])

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]
    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]
    sort!(dtrain, :grp)
    sort!(deval, :grp)

    arch = NeuroTabModels.MLPAttnConfig(; hidden_size=32, nheads=4, stack_size=1, n_attn_layers=1)
    learner = NeuroTabRegressor(arch; loss=:mse, nrounds=20, early_stopping_rounds=5, lr=1e-2, batchsize=64)

    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, deval, group_name="grp", print_every_n=5)

    p = m(deval)
    @test size(p, 1) == nrow(deval)
    @test !any(isnan, p)
end

@testset "Regression - grouped inference row order" begin
    Random.seed!(123)
    nobs = 200
    X = randn(Float32, nobs, 4)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, nobs)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    df[!, :grp] = repeat(1:10, inner=20)
    target_name = "y"
    feature_names = setdiff(names(df), [target_name, "grp"])

    dtrain = sort(df[1:160, :], :grp)
    # the caller's frame is deliberately not in group order
    deval = df[161:end, :][randperm(40), :]

    arch = NeuroTabModels.MLPConfig(; hidden_size=32)
    learner = NeuroTabRegressor(arch; loss=:mse, nrounds=40, lr=1e-2, batchsize=32)
    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, group_name="grp")

    p = m(deval)
    @test size(p, 1) == nrow(deval)
    # predictions must vary, or a misalignment would compare equal anyway
    @test std(p) > 0.5
    # one row per call cannot be reordered by grouping, so it pins each row's own prediction
    p_row = [m(deval[i:i, :])[1] for i in 1:nrow(deval)]
    @test maximum(abs.(p .- p_row)) < 1e-5
end

@testset "Regression - seed makes a fit reproducible" begin
    Random.seed!(123)
    nobs = 200
    X = randn(Float32, nobs, 4)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, nobs)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    df[!, :grp] = repeat(1:10, inner=20)
    target_name = "y"
    feature_names = setdiff(names(df), [target_name, "grp"])

    arch = NeuroTabModels.MLPConfig(; hidden_size=16)
    function fit_predict(seed, global_seed; group_name=nothing)
        # a different global stream must not change the fit, only `seed` may
        Random.seed!(global_seed)
        learner = NeuroTabRegressor(arch; loss=:mse, nrounds=5, lr=1e-2, batchsize=32, seed)
        m = NeuroTabModels.fit(learner, df; target_name, feature_names, group_name)
        return m(df)
    end

    for group_name in (nothing, "grp")
        p = fit_predict(123, 1; group_name)
        @test p == fit_predict(123, 2; group_name)
        @test p != fit_predict(124, 1; group_name)
    end
end

@testset "Classification - grouped" begin
    Random.seed!(123)
    nobs = 360
    X = randn(Float32, nobs, 4)
    score = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.2f0 .* randn(Float32, nobs)
    df = DataFrame(X, :auto)
    df[!, :class] = categorical(ifelse.(score .> 0.5, "hi", ifelse.(score .< -0.5, "lo", "mid")))
    # unequal group sizes, so the grouped loader pads
    df[!, :grp] = rand(1:12, nobs)
    target_name = "class"
    feature_names = setdiff(names(df), [target_name, "grp"])
    dtrain = df[1:280, :]
    deval = df[281:end, :]

    arch = NeuroTabModels.MLPConfig(; hidden_size=32)
    learner = NeuroTabClassifier(arch; nrounds=40, lr=1e-2, batchsize=32)
    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, deval, group_name="grp")

    p = m(deval)
    @test size(p) == (nrow(deval), 3)
    @test maximum(abs.(sum(p; dims=2) .- 1)) < 1e-5
    @test mean(argmax.(eachrow(p)) .== levelcode.(deval.class)) > 0.6
    # the eval metric is scored on the padded groups, so it must match the rows alone
    mlogloss = mean(-log(p[i, levelcode(deval.class[i])]) for i in 1:nrow(deval))
    @test m.info[:logger][:metrics][:metric][end] ≈ mlogloss rtol = 1e-4
end

@testset "MaskedBatchNorm" begin
    rng = Random.Xoshiro(123)
    l = NeuroTabModels.MaskedBatchNorm(4)
    ps, st = Lux.setup(rng, l)
    st_tr = Lux.trainmode(st)

    x_real = randn(Float32, 4, 3)
    x_pad = hcat(x_real, zeros(Float32, 4, 2))
    valid = [true, true, true, false, false]
    y1, _ = l(x_real, ps, st_tr)
    (y2, _), _ = l((x_pad, valid), ps, st_tr)
    @test y2[:, 1:3] ≈ y1

    st_te = Lux.testmode(st)
    y1e, _ = l(x_real, ps, st_te)
    (y2e, _), _ = l((x_pad, valid), ps, st_te)
    @test y2e[:, 1:3] ≈ y1e

    none = falses(5)
    _, st_none = l((x_pad, none), ps, st_tr)
    @test st_none.running_mean == st_tr.running_mean
    @test st_none.running_var == st_tr.running_var
end

@testset "MLPAttn key-padding mask" begin
    Random.seed!(123)
    rng = Random.Xoshiro(123)
    nfeats, hsize, nheads = 6, 16, 4
    arch = NeuroTabModels.MLPAttnConfig(; hidden_size=hsize, nheads, stack_size=1, dropout=0.0)
    chain = arch(; ins=nfeats, outsize=1)
    ps, st = Lux.setup(rng, chain)
    st = Lux.testmode(st)

    x_real = randn(Float32, nfeats, 3)
    y1, _ = chain(x_real, ps, st)

    x_pad = hcat(x_real, zeros(Float32, nfeats, 2))
    w = reshape(Float32[1, 1, 1, 0, 0], 1, 1, 5)
    y2, _ = chain((x_pad, w), ps, st)

    @test size(y1, 2) == 3
    @test size(y2, 2) == 5
    @test y2[:, 1:3] ≈ y1

    st_tr = Lux.trainmode(st)
    y1t, _ = chain(x_real, ps, st_tr)
    y2t, _ = chain((x_pad, w), ps, st_tr)
    @test y2t[:, 1:3] ≈ y1t
end

@testset "MLPAttn BN encoder and attention" begin
    rng = Random.Xoshiro(123)
    nfeats, hsize, nheads = 6, 16, 4
    arch = NeuroTabModels.MLPAttnConfig(; hidden_size=hsize, nheads, stack_size=1, dropout=0.0)
    chain = arch(; ins=nfeats, outsize=1)
    ps, st = Lux.setup(rng, chain)
    st = Lux.testmode(st)

    @test haskey(ps.blocks.layer_1, :qk_proj)
    @test haskey(ps.blocks.layer_1, :scale)
    @test ps.blocks.layer_1.scale.s ≈ Float32[0.1]
    @test !haskey(ps.blocks.layer_1, :norm)
    @test !haskey(ps.blocks.layer_1, :fuse)
    @test !haskey(ps.blocks.layer_1, :v_proj)

    x = randn(Float32, nfeats, 5)
    z, _ = chain.encoder(x, ps.encoder, st.encoder)
    z_mix, _ = chain.blocks(z, ps.blocks, st.blocks)
    @test size(z_mix) == size(z)

    y, _ = chain(x, ps, st)
    @test size(y) == (1, 5)
    @test !any(isnan, y)
    @test !iszero(y)

    arch_id = NeuroTabModels.MLPAttnConfig(; hidden_size=hsize, nheads, stack_size=1, n_attn_layers=1, attn_scale=0.0f0)
    chain_id = arch_id(; ins=nfeats, outsize=1)
    ps_id, st_id = Lux.setup(rng, chain_id)
    st_id = Lux.testmode(st_id)
    z_id, _ = chain_id.encoder(x, ps_id.encoder, st_id.encoder)
    z_mix_id, _ = chain_id.blocks(z_id, ps_id.blocks, st_id.blocks)
    @test z_mix_id ≈ z_id

    arch0 = NeuroTabModels.MLPAttnConfig(; hidden_size=hsize, nheads, stack_size=1, n_attn_layers=0)
    chain0 = arch0(; ins=nfeats, outsize=1)
    ps0, st0 = Lux.setup(rng, chain0)
    st0 = Lux.testmode(st0)
    y0, _ = chain0(x, ps0, st0)
    @test size(y0) == (1, 5)
    @test !any(isnan, y0)

    w = reshape(Float32[1, 1, 1, 1, 1], 1, 1, 5)
    y0m, _ = chain0((x, w), ps0, st0)
    @test y0m ≈ y0
end

@testset "Regression - NeuroTreeAttn grouped" begin
    Random.seed!(123)
    nobs = 400
    X = randn(Float32, nobs, 8)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, nobs)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    df[!, :grp] = repeat(1:20, inner=20)
    target_name = "y"
    feature_names = setdiff(names(df), [target_name, "grp"])

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]
    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]
    sort!(dtrain, :grp)
    sort!(deval, :grp)

    arch = NeuroTabModels.NeuroTreeAttnConfig(;
        hidden_size=32, nheads=4, stack_size=1, n_attn_layers=1, depth=3, ntrees=8
    )
    learner = NeuroTabRegressor(arch; loss=:mse, nrounds=20, early_stopping_rounds=5, lr=1e-2, batchsize=64)

    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, deval, group_name="grp", print_every_n=5)

    p = m(deval)
    @test size(p, 1) == nrow(deval)
    @test !any(isnan, p)
end

@testset "NeuroTreeAttn key-padding mask" begin
    Random.seed!(123)
    rng = Random.Xoshiro(123)
    nfeats, hsize, nheads = 6, 16, 4
    arch = NeuroTabModels.NeuroTreeAttnConfig(;
        hidden_size=hsize, nheads, stack_size=1, dropout=0.0, depth=3, ntrees=4, attn_scale=0.1f0
    )
    chain = arch(; ins=nfeats, outsize=1)
    ps, st = Lux.setup(rng, chain)
    st = Lux.testmode(st)

    x_real = randn(Float32, nfeats, 3)
    y1, _ = chain(x_real, ps, st)

    x_pad = hcat(x_real, zeros(Float32, nfeats, 2))
    w = reshape(Float32[1, 1, 1, 0, 0], 1, 1, 5)
    y2, _ = chain((x_pad, w), ps, st)

    @test size(y1) == (1, hsize, 3)
    @test size(y2) == (1, hsize, 5)
    @test selectdim(y2, 3, 1:3) ≈ y1

    st_tr = Lux.trainmode(st)
    y1t, _ = chain(x_real, ps, st_tr)
    y2t, _ = chain((x_pad, w), ps, st_tr)
    @test selectdim(y2t, 3, 1:3) ≈ y1t
end

@testset "NeuroTreeAttn encoder is k-channels, not shared routing" begin
    rng = Random.Xoshiro(123)
    nfeats, hsize, nheads, depth, ntrees = 6, 16, 4, 3, 4
    @test 2^depth != hsize
    arch = NeuroTabModels.NeuroTreeAttnConfig(; hidden_size=hsize, nheads, stack_size=1, dropout=0.0, depth, ntrees)
    chain = arch(; ins=nfeats, outsize=1)
    ps, st = Lux.setup(rng, chain)
    tree = chain.encoder[1].layer[1]
    @test tree isa NeuroTabModels.Models.NeuroTrees.NeuroTree
    @test tree.k == hsize
    @test tree.outs == 1
    @test tree.leaves == 2^depth
    @test size(ps.encoder.layer_1.layer_1.p) == (1, 2^depth, ntrees, hsize)
    @test ps.blocks.layer_1.scale.s ≈ Float32[0]

    x = randn(Float32, nfeats, 5)
    z, _ = chain.encoder(x, ps.encoder, Lux.testmode(st).encoder)
    z = z isa Tuple ? z[1] : z
    @test size(z) == (hsize, 5)

    y, _ = chain(x, ps, Lux.testmode(st))
    @test size(y) == (1, hsize, 5)
    @test chain.blocks[1].nheads == hsize
    @test chain.blocks[1].qk_proj isa NoOpLayer
    z_mix, _ = chain.blocks(z, ps.blocks, Lux.testmode(st).blocks)
    @test z_mix ≈ z  # default attn_scale=0 is identity

    # Hidden channels are independent ensembles: perturbing one k-slice of leaf
    # values moves only that encoder channel (attention may mix them afterward).
    ps_e = deepcopy(ps)
    ps_e.encoder.layer_1.layer_1.p[:, :, :, 3] .+= 1
    z2, _ = chain.encoder(x, ps_e.encoder, Lux.testmode(st).encoder)
    z2 = z2 isa Tuple ? z2[1] : z2
    @test z[1:2, :] ≈ z2[1:2, :]
    @test z[4:end, :] ≈ z2[4:end, :]
    @test !(z[3:3, :] ≈ z2[3:3, :])
end

@testset "MOETree router softmax mix" begin
    rng = Random.Xoshiro(123)
    nfeats, n_experts, depth, ntrees, batch = 6, 4, 3, 4, 5
    arch = NeuroTabModels.MOETreeConfig(; k=n_experts, depth, ntrees)
    chain = arch(; ins=nfeats, outsize=1)
    NT = NeuroTabModels.Models.NeuroTrees
    @test chain isa NT.MOETree
    @test chain.router isa NT.NeuroTree
    @test chain.experts isa NT.NeuroTree
    @test chain.router.outs == n_experts
    @test chain.router.k == 1
    @test chain.experts.outs == 1
    @test chain.experts.k == n_experts

    ps, st = Lux.setup(rng, chain)
    x = randn(Float32, nfeats, batch)
    y, _ = chain(x, ps, st)
    @test size(y) == (1, 1, batch)

    r, _ = chain.router(x, ps.router, st.router)
    e, _ = chain.experts(x, ps.experts, st.experts)
    @test size(r) == (n_experts, 1, batch)
    @test size(e) == (1, n_experts, batch)
    wr = exp.(r .- maximum(r; dims=1))
    gates = wr ./ sum(wr; dims=1)
    @test all(sum(gates; dims=1) .≈ 1)
    @test y ≈ sum(e .* permutedims(gates, (2, 1, 3)); dims=2)
end

@testset "Backend/device - reactant is a backend" begin
    @test_throws ErrorException NeuroTabRegressor(; backend=:zygote, device=:reactant)
    @test_throws ErrorException NeuroTabClassifier(; backend=:zygote, device=:reactant)
end

@testset "Backend/device - Regression ($backend, $device)" for (backend, device) in
    [(:enzyme, :cpu), (:zygote, :cpu), (:reactant, :cpu)]
    Random.seed!(123)
    X = randn(Float32, 1000, 10)
    y = X[:, 1] .+ 0.5f0 .* X[:, 2] .+ 0.1f0 .* randn(Float32, 1000)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    target_name = "y"
    feature_names = setdiff(names(df), [target_name])

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    learner = NeuroTabRegressor(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(:depth => 3),
        loss=:mse,
        nrounds=20,
        early_stopping_rounds=2,
        lr=1e-1,
        backend,
        device,
    )

    m = NeuroTabModels.fit(learner, dtrain; target_name, feature_names, deval)

    p = m(deval)
    @test size(p, 1) == nrow(deval)
    @test !any(isnan, p)
    mse_model = mean((p .- deval.y) .^ 2)
    mse_baseline = mean((mean(dtrain.y) .- deval.y) .^ 2)
    @test mse_model < mse_baseline
end

@testset "Backend/device - Classification ($backend, $device)" for (backend, device) in
    [(:enzyme, :cpu), (:zygote, :cpu)]
    Random.seed!(123)
    X, y = @load_crabs
    df = DataFrame(X)
    df[!, :class] = y
    target_name = "class"
    feature_names = setdiff(names(df), [target_name])
    transform!(df, feature_names .=> (x -> (x .- mean(x)) ./ std(x)); renamecols=false)

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    learner = NeuroTabClassifier(;
        arch_name="NeuroTreeConfig",
        arch_config=Dict(:depth => 4),
        embedding_config=Dict(:embedding_type => :batchnorm),
        nrounds=200,
        early_stopping_rounds=5,
        lr=3e-2,
        backend,
        device,
    )

    m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names)

    ptrain = [argmax(x) for x in eachrow(m(dtrain))]
    peval = [argmax(x) for x in eachrow(m(deval))]
    @test mean(ptrain .== levelcode.(dtrain.class)) >= 0.95
    @test mean(peval .== levelcode.(deval.class)) >= 0.95
end

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
end

@testset "Output shape contract" begin
    nfeats, outsize, B = 5, 3, 7
    x = randn(Float32, nfeats, B)
    M = NeuroTabModels.Models
    for arch in (
        M.NeuroTreeConfig(), M.MLPConfig(), M.ResNetConfig(), M.TabMConfig(), M.MOETreeConfig(),
        M.MLPAttnConfig(), M.NeuroTreeAttnConfig(),
    )
        chain = arch(; ins=nfeats, outsize)
        ps, st = Lux.setup(Random.Xoshiro(1), chain)
        y, _ = chain(x, ps, Lux.testmode(st))
        @test ndims(y) in (2, 3) && size(y, 1) == outsize && size(y, ndims(y)) == B
    end
end

@testset "Multiple targets" begin
    Random.seed!(123)
    n = 500
    X = randn(Float32, n, 4)
    df = DataFrame(X, :auto)
    df.y1, df.y2, df.y3 = X[:, 1], X[:, 2] .- X[:, 3], sin.(X[:, 4])
    df.b1, df.b2 = Float32.(X[:, 1] .> 0), Float32.(X[:, 2] .> 0)
    df.c1, df.c2 = round.(exp.(0.5f0 .* X[:, 1])), round.(exp.(-0.5f0 .* X[:, 2]))
    df.offset = fill(0.1f0, n)
    feature_names = ["x1", "x2", "x3", "x4"]
    arch = NeuroTabModels.TabMConfig(; k=4, d_block=32, n_blocks=1, dropout=0.0)
    fit_mt(loss, target_name; kw...) = NeuroTabModels.fit(
        NeuroTabRegressor(arch; loss, nrounds=3, lr=1e-2, batchsize=128), df;
        feature_names, target_name, deval=df, kw...,
    )

    @test size(fit_mt(:mse, ["y1", "y2", "y3"])(df)) == (n, 3)
    # Mixed column types promote to one concrete element type.
    df.yint = round.(Int, 10 .* df.y1)
    @test isconcretetype(eltype(fit_mt(:mse, ["yint", "y2"])(df)))

    # Gaussian columns interleave per target, as in EvoTrees: μ₁, σ₁, μ₂, σ₂.
    p = fit_mt(:gaussian_mle, ["y1", "y2"])(df)
    @test size(p) == (n, 4)
    @test all(>(0), p[:, 2:2:end])
    # each pair is unscaled by its own target's scaler: raw (μ, log-σ) per target
    NI, G = NeuroTabModels.Infer, NeuroTabModels.Losses.GaussianMLE()
    raw = Float32[1 0 2 log(2)]
    @test NI._scaler(G, NI._inverse_link(G, raw), (mu=Float32[10, 20], sigma=Float32[2, 3])) ≈ Float32[12 2 26 6]

    @test all(x -> 0 < x < 1, fit_mt(:logloss, ["b1", "b2"])(df))

    # Log link with one offset shared by both targets.
    @test all(>(0), fit_mt(:tweedie, ["c1", "c2"]; offset_name="offset")(df))

    # One offset per target: `e_t = exp(o_t)` is fully explained by its own offset, so predictions,
    # which leave the offset out, stay near 1 only if each target is paired with its own column.
    df.o1, df.o2 = fill(1.5f0, n), fill(-1.5f0, n)
    df.e1, df.e2 = exp.(df.o1), exp.(df.o2)
    p = NeuroTabModels.fit(
        NeuroTabRegressor(arch; loss=:tweedie, nrounds=5, lr=1e-2, batchsize=128), df;
        feature_names, target_name=["e1", "e2"], offset_name=["o1", "o2"],
    )(df)
    @test all(x -> 0.8 < x < 1.25, p)

    # One offset per output for Gaussian, in the order of the predictions; the loss and the
    # eval metric add each column to its own row, a log-σ row included.
    MT = NeuroTabModels.Metrics
    idg = (x, ps, st) -> (x, st)
    pg, og = randn(Float32, 4, 1, 8), randn(Float32, 4, 8)
    yg, wg = randn(Float32, 2, 8), rand(Float32, 8) .+ 0.5f0
    @test first(G(idg, nothing, nothing, (pg, yg, wg, og))) ≈
          first(G(idg, nothing, nothing, (pg .+ reshape(og, 4, 1, 8), yg, wg)))
    pg2 = dropdims(pg; dims=2)
    @test MT.gaussian_mle(_ -> pg2, pg2, yg, wg, og) ≈ MT.gaussian_mle(_ -> pg2 .+ og, pg2, yg, wg)
    df.z1, df.z2 = zeros(Float32, n), zeros(Float32, n)
    @test size(fit_mt(:gaussian_mle, ["y1", "y2"]; offset_name=["o1", "z1", "o2", "z2"])(df)) == (n, 4)
    @test size(fit_mt(:gaussian_mle, ["y1", "y2"]; offset_name="offset")(df)) == (n, 4)

    # A 2D-output architecture with weights and an eval set.
    df.w = rand(Float32, n) .+ 0.5f0
    mlp = NeuroTabRegressor(NeuroTabModels.MLPConfig(); nrounds=2, batchsize=128)
    @test size(NeuroTabModels.fit(mlp, df; feature_names, target_name=["y1", "y2"], weight_name="w", deval=df)(df)) == (n, 2)

    # Targets are averaged, not summed: unit weights leave the training loss unchanged, and
    # the eval steps divide by every target of every observation.
    idm = (x, ps, st) -> (x, st)
    pred, y, w = randn(Float32, 2, 1, 8), randn(Float32, 2, 1, 8), ones(Float32, 8)
    L = NeuroTabModels.Losses
    @test first(L.MSE()(idm, nothing, nothing, (pred, y))) ≈ first(L.MSE()(idm, nothing, nothing, (pred, y, w)))
    p2, y2 = dropdims(pred; dims=2), dropdims(y; dims=2)
    CB = NeuroTabModels.Fit.CallBacks
    for d in ((p2, y2), (p2, y2, w))
        num, den = CB._build_eval_step(idm, NeuroTabModels.Metrics.mse, d, nothing, nothing; reactant=false)(d..., nothing, nothing)
        @test num / den ≈ mean((p2 .- y2) .^ 2)
    end

    # Pearson, as in EvoTrees: each target correlates on its own and the loss and the eval
    # metric take the mean over targets. With two outputs per target the metric reads the μ rows.
    MT = NeuroTabModels.Metrics
    P = L.Pearson()
    pr, yr, wr = randn(Float32, 2, 1, 16), randn(Float32, 2, 16), rand(Float32, 16) .+ 0.5f0
    ploss(t) = first(P(idm, nothing, nothing, (pr[t, :, :], yr[t:t, :], wr)))
    @test first(P(idm, nothing, nothing, (pr, yr, wr))) ≈ (ploss(1) + ploss(2)) / 2
    pp = dropdims(pr; dims=2)
    peval(p, y) = /(CB._build_eval_step(idm, MT.pearson, (p, y, wr), nothing, nothing; reactant=false)(p, y, wr, nothing, nothing)...)
    @test peval(pp, yr) ≈ (peval(pp[1:1, :], yr[1:1, :]) + peval(pp[2:2, :], yr[2:2, :])) / 2
    pg = randn(Float32, 4, 16)
    @test MT.pearson(_ -> pg, pg, yr, wr) ≈ MT.pearson(_ -> pg[[1, 3], :], pg, yr, wr)
    @test size(fit_mt(:pearson, ["y1", "y2"])(df)) == (n, 2)

    # Grouped data: each group's target is `(T, 1, bs)`, zero on pads, and fit and eval run.
    dg = DataFrame(x1=Float32[1, 2, 3, 4, 5], y1=Float32[1, 2, 3, 4, 5], y2=Float32[5, 4, 3, 2, 1], g=[1, 1, 1, 2, 2])
    _, yg, _ = first(NeuroTabModels.Data.get_df_loader_train(
        groupby(dg, :g; sort=true); feature_names=[:x1], target_name=[:y1, :y2], batchsize=0, shuffle=false
    ))
    @test size(yg) == (2, 1, 3) && yg[:, 1, :] == Float32[1 2 3; 5 4 3]
    df.g = repeat(1:10; inner=n ÷ 10)
    @test size(fit_mt(:mse, ["y1", "y2"]; group_name="g")(df)) == (n, 2)
    @test size(fit_mt(:pearson, ["y1", "y2"]; eval_group_name="g")(df)) == (n, 2)

    df.cls = categorical(rand(["a", "b"], n))
    clf = NeuroTabClassifier(arch; nrounds=1)
    @test_throws "Multiple targets" NeuroTabModels.fit(clf, df; feature_names, target_name=["cls", "cls"])
    # A one-name vector is a single target, in the eval set too.
    @test size(NeuroTabModels.fit(clf, df; feature_names, target_name=["cls"], deval=df)(df)) == (n, 2)
    @test_throws "3 columns but the model has 2 outputs" fit_mt(:tweedie, ["c1", "c2"]; offset_name=["o1", "o2", "offset"])
    @test_throws "2 columns but the model has 4 outputs" fit_mt(:gaussian_mle, ["y1", "y2"]; offset_name=["o1", "o2"])
    nca = NeuroTabRegressor(NeuroTabModels.ModernNCAConfig(); nrounds=1)
    @test_throws "`ModernNCAConfig`" NeuroTabModels.fit(nca, df; feature_names, target_name=["y1", "y2"])
    @test_throws "duplicate" fit_mt(:mse, ["y1", "y1"])
    @test_throws "`x1` is also listed in `feature_names`" fit_mt(:mse, ["y1", "x1"])
    df.flat = fill(1.0f0, n)
    @test_throws "`flat` is constant" fit_mt(:mse, ["y1", "flat"])
    df.gap = [i == 7 ? NaN32 : 0.5f0 * i for i in 1:n]
    @test_throws "`gap` has missing or NaN" fit_mt(:mse, ["y1", "gap"])
end

@testset "Pearson loss and metric" begin
    L = NeuroTabModels.Losses
    M = NeuroTabModels.Metrics
    idm = (x, ps, st) -> (x, st)
    pred_fn = x -> x

    x = Float32[1.0 2.0 3.0 4.0]
    y = Float32[1.0 2.0 3.0 4.0]
    val, _, _ = L.Pearson()(idm, (;), (;), (x, y))
    @test val ≈ -1.0f0
    @test M.pearson(pred_fn, x, y) ≈ 4.0f0
    @test M.is_maximise(M.pearson)

    x3 = reshape(x, 1, 1, 4)
    val3, _, _ = L.Pearson()(idm, (;), (;), (x3, y))
    @test val3 ≈ -1.0f0

    w = Float32[1, 1, 1, 1]
    valw, _, _ = L.Pearson()(idm, (;), (;), (x, y, w))
    @test valw ≈ -1.0f0
    @test M.pearson(pred_fn, x, y, w) ≈ 4.0f0

    # Loss is -cor (mean-scale, like MSE). Metric is cor * n so get_metric can average batches.
    x_imp = Float32[1.0 2.0 3.0 4.0]
    y_imp = Float32[1.0 3.0 2.0 8.0]
    c = cor(vec(Float64.(x_imp)), vec(Float64.(y_imp)))
    val_imp, _, _ = L.Pearson()(idm, (;), (;), (x_imp, y_imp))
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

    g = Zygote.gradient(xx -> first(L.Pearson()(idm, (;), (;), (xx, y_imp))), x_imp)
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
    ws = 0.0
    for (x, yb, w) in loader
        mask = vec(w) .> 0
        @test size(x, 2) == 4  # padded to max group size
        @test count(mask) == sum(w)
        p_real = Float64.(x[1, mask])
        y_real = Float64.(vec(yb)[mask])
        n = sum(mask)
        c = cor(p_real, y_real)
        push!(group_cors, c)
        push!(group_ns, n)
        val, _, _ = L.Pearson()(idm, (;), (;), (x, yb, w))
        @test val ≈ -c rtol = 1e-5
        mval = M.pearson(pred_fn, x, yb, w)
        @test mval ≈ c * n rtol = 1e-5
        metric_acc += mval
        ws += sum(w)
    end
    grouped_metric = metric_acc / ws
    @test group_ns == [3.0, 4.0, 4.0]
    @test grouped_metric ≈ sum(group_cors .* group_ns) / sum(group_ns) rtol = 1e-5
    global_cor = cor(Float64.(df.x1), Float64.(df.y))
    @test abs(grouped_metric - global_cor) > 1e-3
    @test -1 <= grouped_metric <= 1
end

@testset "Pearson on flat groups" begin
    L = NeuroTabModels.Losses
    M = NeuroTabModels.Metrics
    idm = (x, ps, st) -> (x, st)
    pred_fn = x -> x
    y = Float32[1.0 3.0 2.0 8.0]
    w = Float32[1, 1, 1, 1]

    # A constant prediction scores 0 instead of 0/0, and its gradient still points along y.
    for c in (0.0f0, 0.3f0)
        p = fill(c, 1, 4)
        val, _, _ = L.Pearson()(idm, (;), (;), (p, y, w))
        @test val ≈ 0 atol = 1e-5
        @test M.pearson(pred_fn, p, y, w) ≈ 0 atol = 1e-5
        g = Zygote.gradient(pp -> first(L.Pearson()(idm, (;), (;), (pp, y, w))), p)[1]
        @test all(isfinite, g)
        @test cor(vec(-g), vec(y)) > 0.99
    end

    # A flat target or a single row scores 0 as well.
    p = Float32[1.0 2.0 3.0 4.0]
    yflat = fill(2.0f0, 1, 4)
    val, _, _ = L.Pearson()(idm, (;), (;), (p, yflat, w))
    @test val ≈ 0 atol = 1e-5
    @test M.pearson(pred_fn, p, yflat, w) ≈ 0 atol = 1e-5
    g = Zygote.gradient(pp -> first(L.Pearson()(idm, (;), (;), (pp, yflat, w))), p)[1]
    @test all(isfinite, g)
    @test M.pearson(pred_fn, Float32[0.5;;], Float32[2.0;;], Float32[1]) ≈ 0 atol = 1e-5
end

@testset "Pearson from a flat start" begin
    # Zero leaves make every prediction equal at iteration 0. A NaN first eval would become the
    # best metric, and early stopping would then fire with best_iter 0.
    Random.seed!(123)
    df = DataFrame(randn(Float32, 40 * 16, 4), :auto)
    df[!, :y] = df.x1 .+ 0.15f0 .* randn(Float32, nrow(df))
    df[!, :grp] = repeat(1:40, inner=16)
    dtrain = df[df.grp.<=32, :]
    deval = df[df.grp.>32, :]
    feature_names = ["x1", "x2", "x3", "x4"]

    arch = NeuroTabModels.NeuroTreeConfig(; depth=3, ntrees=8, stack_size=1, init_scale=0.0)
    for (loss, backend) in ((:mse, :zygote), (:pearson, :zygote), (:pearson, :enzyme), (:pearson, :reactant))
        learner = NeuroTabRegressor(
            arch; loss, metric=:pearson, nrounds=12, early_stopping_rounds=4, lr=1e-2,
            batchsize=0, backend, device=:cpu,
        )
        m = NeuroTabModels.fit(learner, dtrain; target_name="y", feature_names, deval, group_name="grp")
        metrics = m.info[:logger][:metrics][:metric]
        @test all(isfinite, metrics)
        @test first(metrics) ≈ 0 atol = 1e-5
        @test m.info[:logger][:best_iter] > 0
        @test maximum(metrics) > 0.8
        @test !any(isnan, m(deval))
    end

    # One date with a flat target no longer turns the whole grouped metric NaN. It scores 0 and
    # keeps its weight, so it scales the metric by a fixed factor and leaves best_iter unchanged.
    deval_flat = copy(deval)
    deval_flat[deval_flat.grp.==40, :y] .= 1.0f0
    learner = NeuroTabRegressor(
        NeuroTabModels.MLPConfig(; hidden_size=16, stack_size=1, dropout=0.0);
        loss=:mse, metric=:pearson, nrounds=6, early_stopping_rounds=6, lr=1e-2,
        batchsize=0, backend=:zygote, device=:cpu,
    )
    m = NeuroTabModels.fit(learner, dtrain; target_name="y", feature_names, deval=deval_flat, group_name="grp")
    @test all(isfinite, m.info[:logger][:metrics][:metric])
end

@testset "Pearson fit with group_name / eval_group_name" begin
    Random.seed!(123)
    function _pearson_df(n_groups, n_per; shift=0)
        nobs = n_groups * n_per
        X = randn(Float32, nobs, 4)
        y = X[:, 1] .+ 0.15f0 .* randn(Float32, nobs)
        df = DataFrame(X, :auto)
        df[!, :y] = y
        df[!, :grp] = repeat((1:n_groups) .+ shift, inner=n_per)
        return df
    end
    dtrain = _pearson_df(8, 16)
    deval = _pearson_df(4, 12; shift=100)
    target_name = "y"
    feature_names = setdiff(names(dtrain), [target_name, "grp"])

    arch = NeuroTabModels.MLPConfig(; hidden_size=16, stack_size=1, dropout=0.0)
    learner = NeuroTabRegressor(
        arch;
        loss=:pearson,
        metric=:pearson,
        nrounds=8,
        early_stopping_rounds=8,
        lr=3e-2,
        batchsize=32,
        backend=:zygote,
        device=:cpu,
    )

    m_grp = NeuroTabModels.fit(
        learner, dtrain; target_name, feature_names, deval, group_name="grp", print_every_n=8
    )
    metrics_grp = m_grp.info[:logger][:metrics][:metric]
    @test m_grp.info[:group_name] === :grp
    @test m_grp.info[:eval_group_name] === :grp
    @test all(isfinite, metrics_grp)
    @test all(-1.0001 .<= metrics_grp .<= 1.0001)
    @test last(metrics_grp) > first(metrics_grp)

    p = m_grp(deval)
    @test size(p, 1) == nrow(deval)
    @test !any(isnan, p)

    # Ungrouped training, grouped eval metrics (eval_group_name independent of group_name).
    m_eval = NeuroTabModels.fit(
        learner, dtrain; target_name, feature_names, deval, eval_group_name="grp", print_every_n=8
    )
    metrics_eval = m_eval.info[:logger][:metrics][:metric]
    @test isnothing(m_eval.info[:group_name])
    @test m_eval.info[:eval_group_name] === :grp
    @test all(isfinite, metrics_eval)
    @test all(-1.0001 .<= metrics_eval .<= 1.0001)
    @test last(metrics_eval) > first(metrics_eval)
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
    @test_throws "positive and finite" NeuroTabModels.Data.get_df_loader_train(
        groupby(bad, :grp); feature_names=[:x1], target_name=:y, weight_name=:w, batchsize=0
    )

    # and they reach the fit. Odd rows follow y = 2 x1 and even rows y = -x1, so weighting odd
    # rows 9 to 1 gives a least-squares slope of 1.7 against 0.5 unweighted
    Random.seed!(123)
    nobs = 400
    X = randn(Float32, nobs, 4)
    odd = isodd.(1:nobs)
    y = ifelse.(odd, 2 .* X[:, 1], .-X[:, 1]) .+ 0.1f0 .* randn(Float32, nobs)
    d = DataFrame(X, :auto)
    d[!, :y] = y
    d[!, :w] = Float32.(ifelse.(odd, 9, 1))
    d[!, :grp] = repeat(1:20, inner=20)
    feature_names = ["x1", "x2", "x3", "x4"]
    arch = NeuroTabModels.MLPConfig(; hidden_size=32)
    learner = NeuroTabRegressor(arch; loss=:mse, nrounds=40, lr=1e-2, batchsize=32)
    x1 = Float64.(X[:, 1])
    slope(p) = sum((p .- mean(p)) .* (x1 .- mean(x1))) / sum((x1 .- mean(x1)) .^ 2)
    pw = Float64.(NeuroTabModels.fit(learner, d; target_name="y", feature_names, group_name="grp", weight_name="w")(d))
    pu = Float64.(NeuroTabModels.fit(learner, d; target_name="y", feature_names, group_name="grp")(d))
    # over data seeds 1 to 5: 1.44 to 1.69 weighted, 0.18 to 0.55 unweighted
    @test slope(pw) > 1.2
    @test slope(pw) > slope(pu) + 0.5
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

    # and they reach the fit and its eval. The target is x1 + 2 x2 and the offset carries 2 x2,
    # so with it the model is left to learn x1 alone
    Random.seed!(123)
    nobs = 400
    X = randn(Float32, nobs, 4)
    d = DataFrame(X, :auto)
    d[!, :off] = 2 .* X[:, 2]
    d[!, :y] = X[:, 1] .+ d.off .+ 0.1f0 .* randn(Float32, nobs)
    d[!, :grp] = repeat(1:20, inner=20)
    feature_names = ["x1", "x2", "x3", "x4"]
    arch = NeuroTabModels.MLPConfig(; hidden_size=32)
    learner = NeuroTabRegressor(arch; loss=:mse, metric=:pearson, nrounds=40, lr=1e-2, batchsize=32, scale_target=false)
    x2 = Float64.(X[:, 2])
    slope(p) = sum((p .- mean(p)) .* (x2 .- mean(x2))) / sum((x2 .- mean(x2)) .^ 2)
    mo = NeuroTabModels.fit(learner, d; target_name="y", feature_names, group_name="grp", offset_name="off", deval=d)
    po = Float64.(mo(d))
    pn = Float64.(NeuroTabModels.fit(learner, d; target_name="y", feature_names, group_name="grp")(d))
    # slope on x2 over data seeds 1 to 12: -0.04 to 0.07 with the offset, 1.84 to 2.16 without
    @test abs(slope(po)) < 0.5
    @test slope(pn) > 1.5
    # the eval metric is the mean over groups of the correlation of prediction plus offset with y
    r = [cor(po[g] .+ d.off[g], Float64.(d.y[g])) for g in ((20 * (i - 1) + 1):(20 * i) for i in 1:20)]
    @test last(mo.info[:logger][:metrics][:metric]) ≈ mean(r) rtol = 1e-4
end
