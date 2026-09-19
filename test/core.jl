@testset "Core - data iterators" begin end

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
    ("MLPAttn", NeuroTabModels.MLPAttnConfig(; hidden_size=32, nheads=1, stack_size=1, dropout=0.5)),
    (
        "NeuroTreeAttn",
        NeuroTabModels.NeuroTreeAttnConfig(; hidden_size=8, nheads=1, depth=3, ntrees=4, dropout=0.2, init_scale=10),
    ),
    ("ResNet", NeuroTabModels.ResNetConfig(; hidden_size=32, stack_size=1, dropout=0.5)),
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
    pred = Float32[0.2 0.3; 0.0 1.0]
    p = NeuroTabModels.Infer._inverse_link(NeuroTabModels.Losses.GaussianMLE(), pred)
    @test size(p) == (2, 2)
    @test p[:, 1] ≈ Float32[0.2, 0.3]
    @test p[:, 2] ≈ exp.(Float32[0.0, 1.0])

    scalers = (mu=10.0f0, sigma=2.0f0)
    p_scaled = NeuroTabModels.Infer._scaler(NeuroTabModels.Losses.GaussianMLE(), copy(p), scalers)
    @test p_scaled[:, 1] ≈ p[:, 1] .* 2 .+ 10
    @test p_scaled[:, 2] ≈ p[:, 2] .* 2
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
