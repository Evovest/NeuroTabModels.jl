"""
    MOETree

Mixture of NeuroTree experts with a tree router.

The router is one NeuroTree ensemble (`k = 1`) with `outs = N` logits. Softmax
over that prediction axis yields a weight for each of the `N` experts. The
experts are a classical NeuroTree with `k = N` independent ensembles. The
mixture is a weighted sum over the ensemble axis, reducing `k` to 1.

Shapes (`N` experts, `P` outputs, batch `B`):
- router: `(N, 1, B)` → softmax over dim 1
- experts: `(P, N, B)`
- output: `(P, 1, B)`
"""
struct MOETree{R,E} <: LuxCore.AbstractLuxContainerLayer{(:router, :experts)}
    router::R
    experts::E
end

function (m::MOETree)(x::AbstractArray, ps, st)
    r, st_r = m.router(x, ps.router, st.router)
    e, st_e = m.experts(x, ps.experts, st.experts)
    gates = softmax(r; dims=1)
    y = sum(e .* permutedims(gates, (2, 1, 3)); dims=2)
    return y, (; router=st_r, experts=st_e)
end

"""
    MOETreeConfig(; kwargs...)

Mixture of `k` NeuroTree experts gated by a softmax tree router.

Both branches see the same (embedded) features. The router is
`NeuroTree(ins => k; k = 1)` — one ensemble whose leaf predictions are the
`k` logits. Softmax over that axis produces the mixture weights. The experts
are `NeuroTree(ins => outsize; k = k)` — `k` independent ensembles, mixed by
those weights.

# Arguments
- `tree_type::Symbol`: `:binary` or `:oblivious` (default `:binary`).
- `actA::Symbol`: Feature activation on split weights. One of `:identity`, `:tanh`,
  `:hardtanh`, or `:tanhshrink` (default `:identity`).
- `depth::Int`: Tree depth (default `4`).
- `ntrees::Int`: Number of trees averaged in each ensemble (default `32`).
- `k::Int`: Number of experts (default `4`). Router `outs` and expert ensemble
  width. Must be ≥ 1.
- `scaler::Bool`: Apply softplus scaling on tree logits (default `true`).
- `init_scale::Float32`: Leaf weight init scale (default `0.1`).
- `MLE_tree_split::Bool`: Split output head for Gaussian MLE (default `false`).
"""
struct MOETreeConfig <: Architecture
    tree_type::Symbol
    actA::Symbol
    depth::Int
    ntrees::Int
    k::Int
    scaler::Bool
    init_scale::Float32
    MLE_tree_split::Bool
end

function MOETreeConfig(; kwargs...)
    args = Dict{Symbol,Any}(
        :tree_type => :binary,
        :actA => :identity,
        :depth => 4,
        :ntrees => 32,
        :k => 4,
        :scaler => true,
        :init_scale => 0.1,
        :MLE_tree_split => false,
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    length(args_ignored) > 0 &&
        @warn "Following $(length(args_ignored)) provided arguments will be ignored: $(join(args_ignored, ", "))."

    args_default = setdiff(keys(args), keys(kwargs))
    length(args_default) > 0 &&
        @info "Following $(length(args_default)) arguments set to default: $(join(args_default, ", "))."

    for arg in intersect(keys(args), keys(kwargs))
        args[arg] = kwargs[arg]
    end

    return MOETreeConfig(
        Symbol(args[:tree_type]),
        Symbol(args[:actA]),
        args[:depth],
        args[:ntrees],
        args[:k],
        args[:scaler],
        args[:init_scale],
        args[:MLE_tree_split],
    )
end

function _moe_tree_kwargs(config::MOETreeConfig)
    return (;
        tree_type=config.tree_type,
        depth=config.depth,
        trees=config.ntrees,
        actA=act_dict[config.actA],
        scaler=config.scaler,
        init_scale=config.init_scale,
    )
end

function _build_moe_tree(ins::Int, outsize::Int, config::MOETreeConfig)
    n_experts = config.k
    n_experts >= 1 || error("`k` (number of experts) must be ≥ 1, got $n_experts.")
    kwargs = _moe_tree_kwargs(config)
    router = NeuroTree(ins => n_experts; k=1, kwargs...)
    experts = NeuroTree(ins => outsize; k=n_experts, kwargs...)
    return MOETree(router, experts)
end

"""
    (config::MOETreeConfig)(; ins, outsize)

Build a [`MOETree`](@ref) backbone from `config`.
"""
function (config::MOETreeConfig)(; ins, outsize, kwargs...)
    if config.MLE_tree_split
        iseven(outsize) || error("MLE_tree_split requires an even `outsize` (e.g., 2 for μ and σ). Got: $outsize")
        head_outsize = outsize ÷ 2
        return Chain(
            Parallel(vcat, _build_moe_tree(ins, head_outsize, config), _build_moe_tree(ins, head_outsize, config))
        )
    end
    return _build_moe_tree(ins, outsize, config)
end
