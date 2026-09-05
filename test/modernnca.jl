# Dense reference: full softmax over all keys, no chunking.
function dense_nca(model, zq, zk, y; mask_self=false)
    modernnca = NeuroTabModels.Models.ModernNCA
    _, s = modernnca._scores(model, zq, zk; mask_self)
    α = exp.(s .- maximum(s; dims=1))
    α = α ./ sum(α; dims=1)
    return modernnca._finalize(model.loss_type,
        modernnca._train_targets(model, y, 1:size(zk, 2)) * α)
end

@testset "ModernNCA integration" begin
    rng = Xoshiro(123)
    x = randn(rng, Float32, 40, 4)
    df = DataFrame(x, :auto)
    df.y = x[:, 1] .- 0.5f0 .* x[:, 2]
    dtrain, deval = df[1:32, :], df[33:end, :]
    features = names(df, r"x")

    arch = NeuroTabModels.ModernNCAConfig(;
        d_embedding=8, n_blocks=0, sample_rate=0.5,
        corpus_chunk_size=7)
    learner = NeuroTabRegressor(arch;
        embedding_config=NeuroTabModels.LinearEmbeddings(; d_embedding=2),
        nrounds=5, early_stopping_rounds=5, batchsize=16,
        scale_target=false, backend=:zygote, device=:cpu)
    model = NeuroTabModels.fit(
        learner, dtrain;
        feature_names=features, target_name=:y, deval, verbosity=0)

    @test size(model.info[:nca_ref].cx, 2) == nrow(dtrain)
    prediction = model(deval)
    @test all(isfinite, prediction)
    metrics = model.info[:logger][:metrics][:metric]
    @test length(unique(metrics)) > 1
    @test sum(abs2, prediction .- deval.y) / nrow(deval) ≈
          last(metrics) rtol=1f-5

    deval.group = repeat(1:2; inner=4)
    model.info[:group_key] = :group
    @test length(model(deval)) == nrow(deval)
    select!(deval, Not(:group))

    arch = NeuroTabModels.ModernNCAConfig(; d_embedding=8, n_blocks=0)
    @test_throws ArgumentError arch(
        ; ins=4, outsize=2,
        loss_type=NeuroTabModels.Losses.GaussianMLE)
    @test_throws ArgumentError NeuroTabModels.ModernNCAConfig(
        ; corpus_chunk_size=0)

    model = arch(
        ; ins=4, outsize=1, loss_type=NeuroTabModels.Losses.MSE)
    @test_throws ArgumentError NeuroTabModels.Models.infer_dataloader(
        model, Dict(), (), identity, nothing, nothing; backend=:reactant)

    fitted = NeuroTabModels.Models.NeuroTabModel(
        NeuroTabModels.Losses.MSE, model, Dict{Symbol,Any}())
    for unsupported in
        ((; weight_name=:w), (; offset_name=:o), (; group_key=:g))
        @test_throws ArgumentError NeuroTabModels.Models.train_dataloader(
            arch, fitted, nothing, dtrain;
            feature_names=features, target_name=:y,
            loss_type=NeuroTabModels.Losses.MSE, scalers=nothing,
            batchsize=8, dev=identity, rng, unsupported...)
    end
end

@testset "ModernNCA chunked equivalence" begin
    rng = Xoshiro(42)
    modernnca = NeuroTabModels.Models.ModernNCA
    Lux = modernnca.Lux

    for (loss_type, outsize, make_y) in (
        (NeuroTabModels.Losses.MSE, 1, n -> randn(rng, Float32, n)),
        (NeuroTabModels.Losses.LogLoss, 1,
            n -> Float32.(rand(rng, 0:1, n))),
        (NeuroTabModels.Losses.MLogLoss, 5,
            n -> UInt32.(rand(rng, 1:5, n))),
    )
        model = NeuroTabModels.ModernNCAConfig(;
            d_embedding=16, n_blocks=0, corpus_chunk_size=777)(
            ; ins=16, outsize, loss_type)
        ps, st = Lux.setup(rng, model)
        st = Lux.testmode(st)
        y = make_y(5000)
        x = randn(rng, Float32, 16, 64)
        corpus = modernnca.Corpus(
            randn(rng, Float32, 16, 5000),
            modernnca._target_layout(loss_type, y), Dict(:nrounds => 0))

        chunked, _ = model((x, corpus), ps, st)
        zq, _ = modernnca._encode(model, x, ps, st)
        dense = dense_nca(model, zq, modernnca._keys(model, corpus, ps, st), y)
        @test chunked ≈ dense rtol=1f-4
    end

    info = Dict{Symbol,Any}(:nrounds => 0)
    mse_model = NeuroTabModels.ModernNCAConfig(; d_embedding=4, n_blocks=0)(
        ; ins=3, outsize=1, loss_type=NeuroTabModels.Losses.MSE)
    ps, st = Lux.setup(Xoshiro(7), mse_model)
    st = Lux.testmode(st)
    corpus = modernnca.Corpus(
        randn(rng, Float32, 3, 12), randn(rng, Float32, 1, 12), info)
    z1 = modernnca._keys(mse_model, corpus, ps, st)
    @test modernnca._keys(mse_model, corpus, ps, st) === z1
    info[:nrounds] = 1
    @test modernnca._keys(mse_model, corpus, ps, st) !== z1

    loader = modernnca.ModernNCALoader(
        randn(rng, Float32, 3, 20), randn(rng, Float32, 20),
        4, 10, rng, identity)
    (_, cand_x, _, _), _ = first(loader)
    @test size(cand_x, 2) == 10
    @test size(unique(cand_x; dims=2), 2) == 10

    x = randn(rng, Float32, 3, 2)
    moved_x, retained_corpus = Lux.cpu_device()((x, corpus))
    @test moved_x == x
    @test retained_corpus === corpus
end

@testset "ModernNCA chunked training gradient" begin
    rng = Xoshiro(7)
    modernnca = NeuroTabModels.Models.ModernNCA
    Lux = modernnca.Lux

    for (loss_type, outsize, make_y) in (
            (NeuroTabModels.Losses.MSE, 1, n -> randn(rng, Float32, n)),
            (NeuroTabModels.Losses.MLogLoss, 3, n -> UInt32.(rand(rng, 1:3, n))),
        ),
        (n_blocks, chunk) in ((0, 13), (1, 1000))  # BatchNorm only with a single chunk

        model = NeuroTabModels.ModernNCAConfig(;
            d_embedding=8, n_blocks, d_block=16, dropout=0.0,
            corpus_chunk_size=chunk)(; ins=6, outsize, loss_type)
        ps, st = Lux.setup(rng, model)
        x, cand_x = randn(rng, Float32, 6, 12), randn(rng, Float32, 6, 100)
        y, cand_y = make_y(12), make_y(100)

        dense = ps_ -> begin
            zq, st1 = modernnca._encode(model, x, ps_, st)
            zc, _ = modernnca._encode(model, cand_x, ps_, st1)
            sum(dense_nca(model, zq, hcat(zq, zc), vcat(y, cand_y); mask_self=true))
        end
        chunked = ps_ -> sum(first(model((x, cand_x, cand_y, y), ps_, st)))

        @test chunked(ps) ≈ dense(ps) rtol=1f-4
        gd = Zygote.gradient(dense, ps)[1]
        gc = Zygote.gradient(chunked, ps)[1]
        for (a, b) in zip(modernnca.Functors.fleaves(gd), modernnca.Functors.fleaves(gc))
            @test a ≈ b rtol=1f-3
        end
    end
end
