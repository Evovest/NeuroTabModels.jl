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
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

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
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

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
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

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
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

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

    m = NeuroTabModels.fit(learner, dtrain; deval, target_name, feature_names, print_every_n=5);

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
        NeuroTabModels.NeuroTreeAttnConfig(; hidden_size=8, nheads=1, stack_size=1, depth=3, ntrees=8, dropout=0.5),
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
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

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
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]
    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]
    sort!(dtrain, :grp)
    sort!(deval, :grp)

    arch = NeuroTabModels.MLPAttnConfig(; hidden_size=32, nheads=4, stack_size=1, n_attn_layers=1)
    learner = NeuroTabRegressor(arch; loss=:mse, nrounds=20, early_stopping_rounds=5, lr=1e-2, batchsize=64)

    m = NeuroTabModels.fit(
        learner, dtrain; target_name, feature_names, deval, group_key="grp", print_every_n=5
    )

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
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]
    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]
    sort!(dtrain, :grp)
    sort!(deval, :grp)

    arch = NeuroTabModels.NeuroTreeAttnConfig(;
        hidden_size=32, nheads=4, stack_size=1, n_attn_layers=1, depth=3, ntrees=8
    )
    learner = NeuroTabRegressor(arch; loss=:mse, nrounds=20, early_stopping_rounds=5, lr=1e-2, batchsize=64)

    m = NeuroTabModels.fit(
        learner, dtrain; target_name, feature_names, deval, group_key="grp", print_every_n=5
    )

    p = m(deval)
    @test size(p, 1) == nrow(deval)
    @test !any(isnan, p)
end

@testset "NeuroTreeAttn key-padding mask" begin
    Random.seed!(123)
    rng = Random.Xoshiro(123)
    nfeats, hsize, nheads = 6, 16, 4
    arch = NeuroTabModels.NeuroTreeAttnConfig(;
        hidden_size=hsize, nheads, stack_size=1, dropout=0.0, depth=3, ntrees=4
    )
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

@testset "NeuroTreeAttn encoder is k-ensembles, not leaves" begin
    rng = Random.Xoshiro(123)
    nfeats, hsize, nheads, depth, ntrees = 6, 16, 4, 3, 4
    @test 2^depth != hsize
    arch = NeuroTabModels.NeuroTreeAttnConfig(;
        hidden_size=hsize, nheads, stack_size=1, dropout=0.0, depth, ntrees
    )
    chain = arch(; ins=nfeats, outsize=1)
    ps, st = Lux.setup(rng, chain)
    tree = chain.encoder[1].layer[1]
    @test tree isa NeuroTabModels.Models.NeuroTrees.NeuroTree
    @test tree.k == hsize
    @test tree.outs == 1
    @test tree.leaves == 2^depth
    @test size(ps.encoder.layer_1.layer_1.p) == (1, 2^depth, ntrees, hsize)

    x = randn(Float32, nfeats, 5)
    z, _ = chain.encoder(x, ps.encoder, Lux.testmode(st).encoder)
    z = z isa Tuple ? z[1] : z
    @test size(z) == (hsize, 5)
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
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

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
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

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

@testset "Correlation loss" begin
    idm = (x, ps, st) -> (x, st)
    x = Float32[1.0 2.0 3.0 4.0]
    y = Float32[1.0 2.0 3.0 4.0]
    val, _, _ = NeuroTabModels.Losses.Correlation()(idm, (;), (;), (x, y))
    @test val ≈ -4.0f0

    x3 = reshape(x, 1, 1, 4)
    val3, _, _ = NeuroTabModels.Losses.Correlation()(idm, (;), (;), (x3, y))
    @test val3 ≈ -4.0f0

    w = Float32[1, 1, 1, 1]
    valw, _, _ = NeuroTabModels.Losses.Correlation()(idm, (;), (;), (x, y, w))
    @test valw ≈ -4.0f0
end