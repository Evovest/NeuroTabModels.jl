using BenchmarkTools
using Random
using NeuroTabModels
using NeuroTabModels.Models.NeuroTrees
# using CUDA
using CairoMakie

# density(m.chain.layers[2].trees[1].b)
# density(m.chain.layers[2].trees[1].s)
# mean(m.chain.layers[2].trees[1].s)
# density(vec(m.chain.layers[2].trees[1].p))
# density(vec(m.chain.layers[2].trees[1].w))
# density(abs.(vec(m.chain.layers[2].trees[1].w)))
# mean(abs.(vec(m.chain.layers[2].trees[1].w)) .< 1e-1)

###############################
# softplus mask
###############################
mask = NeuroTrees.get_softplus_mask(Val(:binary), 4)
mask = NeuroTrees.get_softplus_mask(Val(:oblivious), 4)

fig = Figure(; size=(450, 450))
ax = Axis(fig[1, 1];
    title="softplus mask",
    xlabel="Leaves",
    ylabel="Nodes",
    aspect=DataAspect(),
    xticks=collect(1:size(mask, 1)),
    yticks=collect(1:size(mask, 2)),
    xgridcolor=:lightgrey,
    ygridcolor=:lightgrey
)
hm = heatmap!(ax, 1:size(mask, 1)+1, 1:size(mask, 2)+1, mask, colormap=[:white, "#1f4e79"])
translate!(hm, 0, 0, -100)
fig

fig = Figure(; size=(900, 450))
mask = NeuroTrees.get_softplus_mask(Val(:binary), 4)
ax = Axis(fig[1, 1];
    title="softplus mask",
    xlabel="Leaves",
    ylabel="Nodes",
    aspect=DataAspect(),
    xticks=collect(1:size(mask, 1)),
    yticks=collect(1:size(mask, 2)),
    xgridcolor=:lightgrey,
    ygridcolor=:lightgrey
)
hm = heatmap!(ax, 1:size(mask, 1)+1, 1:size(mask, 2)+1, mask, colormap=[:white, "#1f4e79"])
translate!(hm, 0, 0, -100)

mask = NeuroTrees.get_softplus_mask(Val(:oblivious), 4)
ax = Axis(fig[1, 2];
    title="softplus mask",
    xlabel="Leaves",
    ylabel="Nodes",
    aspect=DataAspect(),
    xticks=collect(1:size(mask, 1)),
    yticks=collect(1:size(mask, 2)),
    xgridcolor=:lightgrey,
    ygridcolor=:lightgrey
)
hm = heatmap!(ax, 1:size(mask, 1)+1, 1:size(mask, 2)+1, mask, colormap=[:white, "#1f4e79"])
translate!(hm, 0, 0, -100)
fig

###############################
# logits mask
###############################
mask = NeuroTrees.get_logits_mask(Val(:binary), 4)
mask = NeuroTrees.get_logits_mask(Val(:oblivious), 4)

fig = Figure(; size=(450, 450))
ax = Axis(fig[1, 1];
    title="logits mask",
    xlabel="Leaves",
    ylabel="Nodes",
    aspect=DataAspect(),
    xticks=collect(1:size(mask, 1)),
    yticks=collect(1:size(mask, 2)),
    xgridcolor=:lightgrey,
    ygridcolor=:lightgrey
)
hm = heatmap!(ax, 1:size(mask, 1)+1, 1:size(mask, 2)+1, mask, colormap=[:white, "#1f4e79"])
translate!(hm, 0, 0, -100)
fig


fig = Figure(; size=(900, 450))
mask = NeuroTrees.get_logits_mask(Val(:binary), 4)
ax = Axis(fig[1, 1];
    title="logits mask",
    xlabel="Leaves",
    ylabel="Nodes",
    aspect=DataAspect(),
    xticks=collect(1:size(mask, 1)),
    yticks=collect(1:size(mask, 2)),
    xgridcolor=:lightgrey,
    ygridcolor=:lightgrey
)
hm = heatmap!(ax, 1:size(mask, 1)+1, 1:size(mask, 2)+1, mask, colormap=[:white, "#1f4e79"])
translate!(hm, 0, 0, -100)

mask = NeuroTrees.get_logits_mask(Val(:oblivious), 4)
ax = Axis(fig[1, 2];
    title="logits mask",
    xlabel="Leaves",
    ylabel="Nodes",
    aspect=DataAspect(),
    xticks=collect(1:size(mask, 1)),
    yticks=collect(1:size(mask, 2)),
    xgridcolor=:lightgrey,
    ygridcolor=:lightgrey
)
hm = heatmap!(ax, 1:size(mask, 1)+1, 1:size(mask, 2)+1, mask, colormap=[:white, "#1f4e79"])
translate!(hm, 0, 0, -100)
fig


function leaf_weights(logits, lmask, smask)
    logits * lmask .+ softplus.(logits) * smask
end
##############
# test
##############
depth = 3
tree_type = :binary
nw = tree_type == :oblivious ? zeros(depth) : zeros(2^depth - 1)
nw[2] = -1.0
lmask = NeuroTrees.get_logits_mask(Val(tree_type), depth)
smask = NeuroTrees.get_softplus_mask(Val(tree_type), depth)
lw = exp.(lmask * nw .- smask * NeuroTrees.softplus.(nw)) # [N,TB] => [L,TB]
sum(lw)


########################
# benchmarks
########################
function matmul_version(nw, mask)
    mask * nw
end

function broadcast_version(nw, mask)
    # reshape(nw, 1, :) is the same as nw' in row form for broadcasting
    dropdims(sum(mask .* reshape(nw, 1, size(nw)...); dims=2); dims=2)
end

Random.seed!(123)
tree_type = :binary
# depth = 2

for depth in [4, 5, 6]
    println("\n=== tree_type = $tree_type | depth = $depth ===")

    lmask = NeuroTrees.get_logits_mask(Val(tree_type), depth)
    # lmask = lmask |> CuArray
    lmask = Float32.(lmask) # |> CuArray
    # lmask = BitMatrix(lmask) |> CuArray

    nw = randn(Float32, size(lmask, 2), 1024)
    # nw = nw |> CuArray

    # Warmup
    matmul_version(nw, lmask)
    broadcast_version(nw, lmask)

    b_mat = @benchmark matmul_version($nw, $lmask)
    b_bc = @benchmark broadcast_version($nw, $lmask)

    t_mat = median(b_mat.times) / 1e3   # μs
    t_bc = median(b_bc.times) / 1e3    # μs

    println("  Matmul:     $(round(t_mat;  digits=2)) μs   (± $(round(std(b_mat.times)/1e3; digits=2)) μs)")
    println("  Broadcast:  $(round(t_bc;  digits=2)) μs   (± $(round(std(b_bc.times)/1e3;  digits=2)) μs)")
    println("  Ratio bc / matmul = $(round(t_bc / t_mat; digits=3))×")
end
