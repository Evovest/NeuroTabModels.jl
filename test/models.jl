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

@testset "Only MaskedModel keeps the grouped mask" begin
    m = NeuroTabModels.Models.MaskedModel(nothing, nothing)
    x, mask = randn(Float32, 3, 4), trues(4)
    @test NeuroTabModels.Losses.masked_input(m, x) === x
    @test NeuroTabModels.Losses.masked_input(m, (x, mask)) === (x, mask)
    @test NeuroTabModels.Losses.masked_input(nothing, (x, mask)) === x
end

@testset "AttnResidual multi-head attention" begin
    layer = NeuroTabModels.Models.AttnResidual(4, 2; identity_qk=true, attn_scale=1.0f0)
    ps, st = Lux.setup(Random.Xoshiro(123), layer)
    st = Lux.testmode(st)
    x = Float32[1 2 10; 3 1 5; 2 4 8; 1 3 6]

    for valid in (trues(3), Bool[true, true, false])
        mask = reshape(valid, 3, 1, 1, 1)
        y, _ = layer((x, mask), ps, st)
        qk = x ./ sqrt.(sum(abs2, x[:, valid]; dims=2) ./ Float32(sum(valid)) .+ 1.0f-5)
        expected = similar(x)
        for head in 1:2, i in 1:3
            channels = (2head - 1):(2head)
            scores = [sum(qk[channels, j] .* qk[channels, i]) / sqrt(2.0f0) for j in findall(valid)]
            weights = exp.(scores .- maximum(scores))
            weights ./= sum(weights)
            expected[channels, i] = x[channels, valid] * weights
        end
        @test y[1] ≈ x .+ expected
    end
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

    @test size(y1, 3) == 3
    @test size(y2, 3) == 5
    @test y2[:, :, 1:3] ≈ y1

    st_tr = Lux.trainmode(st)
    y1t, _ = chain(x_real, ps, st_tr)
    y2t, _ = chain((x_pad, w), ps, st_tr)
    @test y2t[:, :, 1:3] ≈ y1t
end

@testset "MLPAttn BN encoder and attention" begin
    rng = Random.Xoshiro(123)
    nfeats, hsize, nheads = 6, 16, 4
    arch = NeuroTabModels.MLPAttnConfig(; hidden_size=hsize, nheads, stack_size=1, dropout=0.0)
    chain = arch(; ins=nfeats, outsize=1)
    ps, st = Lux.setup(rng, chain)
    st = Lux.testmode(st)

    @test ps.blocks.layer_1.scale.s ≈ Float32[0.1]

    x = randn(Float32, nfeats, 5)
    z, _ = chain.encoder(x, ps.encoder, st.encoder)
    z_mix, _ = chain.blocks(z, ps.blocks, st.blocks)
    @test size(z_mix) == size(z)

    y, _ = chain(x, ps, st)
    @test size(y) == (1, 1, 5)
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
    @test size(y0) == (1, 1, 5)
    @test !any(isnan, y0)

    w = reshape(Float32[1, 1, 1, 1, 1], 1, 1, 5)
    y0m, _ = chain0((x, w), ps0, st0)
    @test y0m ≈ y0
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
    @test chain.router.init_scale == 0
    @test chain.experts.outs == 1
    @test chain.experts.k == n_experts
    @test chain.experts.init_scale == 1

    ps, st = Lux.setup(rng, chain)
    x = randn(Float32, nfeats, batch)
    y, _ = chain(x, ps, st)
    @test size(y) == (1, 1, batch)
    @test all(iszero, ps.router.p)

    r, _ = chain.router(x, ps.router, st.router)
    @test all(iszero, r)
    e, _ = chain.experts(x, ps.experts, st.experts)
    @test size(r) == (n_experts, 1, batch)
    @test size(e) == (1, n_experts, batch)
    wr = exp.(r .- maximum(r; dims=1))
    gates = wr ./ sum(wr; dims=1)
    @test all(sum(gates; dims=1) .≈ 1)
    @test all(gates .≈ 1 / n_experts)
    @test y ≈ sum(e .* permutedims(gates, (2, 1, 3)); dims=2)
end

@testset "NeuroTree leaf init policy" begin
    NT = NeuroTabModels.Models.NeuroTrees

    single = NT.NeuroTreeConfig(; depth=3, ntrees=4, stack_size=1)(; ins=5, outsize=1)
    @test single.layer_1.chain.init_scale == 0

    stacked = NT.NeuroTreeConfig(; depth=3, ntrees=4, stack_size=2, hidden_size=2)(; ins=5, outsize=1)
    inner = stacked.layer_1.chain
    @test inner.layer_1.init_scale == 1
    @test inner.layer_3.init_scale == 0

    attn = NT.NeuroTreeAttnConfig(; hidden_size=4, depth=2, ntrees=2, stack_size=2)(; ins=5, outsize=1)
    @test attn.encoder.layer_1.layer.layer_1.init_scale == 1
    @test attn.encoder.layer_2.layer.layer.layer_1.init_scale == 1
end

@testset "Output shape contract" begin
    # Every architecture returns `(outsize, k, batch)`, with `k = 1` for a single model.
    nfeats, B = 5, 7
    x = randn(Float32, nfeats, B)
    M = NeuroTabModels.Models
    for arch in (
        M.NeuroTreeConfig(; k=3), M.MLPConfig(), M.ResNetConfig(), M.TabMConfig(; k=3), M.MOETreeConfig(),
        M.MLPAttnConfig(), M.NeuroTreeAttnConfig(), M.MLPConfig(; MLE_tree_split=true),
        M.NeuroTreeConfig(; MLE_tree_split=true, k=3),
    )
        chain = arch(; ins=nfeats, outsize=4)
        ps, st = Lux.setup(Random.Xoshiro(1), chain)
        y, _ = chain(x, ps, Lux.testmode(st))
        k = hasproperty(arch, :k) && !(arch isa M.MOETreeConfig) ? arch.k : 1
        @test size(y) == (4, k, B)
    end
end
