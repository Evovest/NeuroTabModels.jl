# Grouped padding and masks

`MLPAttn` and `NeuroTreeAttn` let information travel across observations that
belong to the same group. A grouped loader (`group_name`) uses `batchsize=0`:
each step is one group, padded to a common width. Attention’s own batch
dimension is therefore always `1` — the sequence is the group.

This page covers that padding: where `w` comes from, why embeddings never see
it, and how the `*Attn` core consumes it.

## Why there is a pad at all

A grouped train/eval/infer loader takes `groupby(df, group_name)` and writes
**one padded tensor per group**. The buffer width is the size of the largest
group, so every group is a dense `(features, buffer)` matrix:

```
one group, 3 real rows, buffer = 5

        col:   1    2    3    4    5
               ├──── real ────┤  pads

   x  =  [  x₁   x₂   x₃   0    0  ]     (nfeats × buffer)
   y  =  [  y₁   y₂   y₃   0    0  ]
   w  =  [  1    1    1    0    0  ]     1 = real row, 0 = pad slot
```

Train/eval use `w` as that 0/1 flag (shape `(1, 1, buffer)`). Infer uses a
boolean `mask` of length `buffer` and, after the forward, keeps only
`pred[:, mask]`.

Pads exist so the group is a rectangular array. They are not observations.
Anything that reduces across the last dimension must ignore them. Per-column
maps (embeddings, Dense heads) can run on pad columns without contaminating
real ones.

## Two jobs for `w`

`w` is reused for two different things, at two different places:

| Job | Who uses it | What happens to pads |
| --- | --- | --- |
| **Loss / metric weight** | `_reduce(loss, w)` and eval | pad slots contribute `0` to the scalar loss |
| **Attention mask** | `MLPAttn` / `NeuroTreeAttn` | pads are excluded from attention keys and from `MaskedBatchNorm` stats |

A non-attention model (`ResNet`, `MLP`, …) only needs the first job. Its
forward sees `x` only; pad rows still produce a prediction, but that
prediction is multiplied by `w = 0` in the loss. Grouped training works for
those models without threading a mask through the net.

An `*Attn` model also needs the second job. A pad token with value `0` would
still be a key every query can attend to. Encoder BatchNorm is a reduction
over the group: pad zeros would enter `μ` / `σ` and running stats. Those
layers must see a valid-token flag.

## Why embeddings do not take `w`

Embeddings are per-observation maps. A linear / periodic / piecewise-linear /
identity / layer-norm embedding applied to column `j` does not read column
`k`. Computing `embed(x_pad)` cannot change `embed(x_real)`.

The mask is therefore not an embedding concern. The split is implemented once,
at the embed / core boundary:

```
input is either x  or  (x, w)

                    (x, w)
                       │
          ┌────────────┴────────────┐
          │      MaskedModel        │
          │                         │
          │   z = embed(x)          │  ← embeddings: x only
          │                         │
          │   y = core((z, w))      │  ← *Attn / MaskedBN: (z, w)
          └────────────┬────────────┘
                       │
                       y
```

`fit` chooses that wrapper from the core, not from the loader:

```julia
chain = if uses_batch_mask(core_chain)
    MaskedModel(embed_chain, core_chain)   # MLPAttn, NeuroTreeAttn
else
    Chain(embed_chain, core_chain)          # everyone else
end
```

`Chain(embed, core)` would pass `(x, w)` into `embed`, which does not implement
that signature. `MaskedModel` peels the tuple: embeddings stay ordinary Lux
layers; only the core is mask-aware.

Call sites do not special-case embeddings either. `masked_input` decides
whether `w` is forwarded into the assembled chain:

```
Chain:         masked_input(model, x, w) = x
MaskedModel:   masked_input(model, x, w) = (x, w)
```

Loss, eval, and infer all go through that hook. The loader always produces
`(x, y, w)` for grouped data; only mask-aware chains unpack `w` as a model
input.

Pad columns are still forwarded. After `embed(x)` the core keeps a rectangular
`(hidden, buffer)` stream: attention and `MaskedBatchNorm` drop pads from
reductions, the head still emits a prediction per slot, and the loss zeros
those slots. Embeddings may produce garbage on pad columns; those tokens must
not leak into real tokens downstream.

`BatchNormEmbeddings` is the exception: it does reduce across the group, and
today it does not see `w`. Prefer identity / linear / LN embeddings (or the
model’s own `MaskedBatchNorm`) when training an `*Attn` model on grouped data.

## End-to-end flow

```
GroupedDataFrame
        │
        ▼
  padded buffers          x  (nfeats × buffer)
                        w/mask  (1 on real, 0/false on pad)
        │
        ▼
  masked_input(chain, x, w)
        │
        ├── Chain:          x                 → embed(x) → core(z)
        └── MaskedModel:    (x, w)            → embed(x) → core(z, w)
        │
        ▼
  y  (outsize × buffer)     pad slots still have a number
        │
        ├── train/eval:  loss weighted by w  (pads drop out)
        └── infer:       pred[:, mask]       (pads dropped after forward)
```

## Inside MLPAttn / NeuroTreeAttn

`MLPAttn` converts `w` to a boolean `valid` once, then uses the same flags in
two shapes.

```
(x, w)
   │
   ├─ valid = vec(w) .> 0          length = buffer
   │
   ▼
encoder  ←  (x, valid)
   │
   │   residual MLP, same recipe as ResNet, but BN is masked:
   │
   │   CarryMask(Dense)     (h, valid) → (Dense(h), valid)
   │   MaskedBatchNorm      μ, σ from columns where valid is true
   │   CarryMask(Dropout)
   │   MaskSkip             (h, valid) → (h + f(h), valid)
   │
   ▼
z, drop the tuple
   │
   ▼
AttnResidual  ←  (z, mask)
   │
   │   mask = reshape(valid, buffer, 1, 1, 1)
   │   Q = K = W_qk z ,   V = z
   │   attention batch dim = 1  (the group is the sequence)
   │   rectangular key-padding mask (not causal):
   │   padded keys are zeroed for every query
   │
   ▼
head(z)        Dense on every column, including pads
```

`CarryMask` / `MaskSkip` exist so a `Chain` of ordinary layers can thread
`(features, mask)` without each `Dense` knowing about padding.
`MaskedBatchNorm` is the layer that actually reads the flag.

`NeuroTreeAttn` is the same split with a tree encoder: trees are
per-observation (`CarryMask` around `NeuroTree` + flatten), then
`MaskedBatchNorm` uses the valid-token flag like MLPAttn. Attention blocks
receive a rectangular key-padding mask. The hidden width is NeuroTree `k`
(one ensemble of `ntrees` per channel, scalar leaf preds), not the number of
leaves.

The mask is a key-padding mask over observations in the group, not a feature
mask and not a causal triangle.

## Ungrouped batches

With no `group_name`, there are no pad slots. Every column is a real
observation. `*Attn` models still accept a plain `x` (the unmasked method).
`n_attn_layers=0` on `MLPAttn` is that encoder plus a linear head: a ResNet
ablation, including `MaskedBatchNorm` with `valid = nothing`, which is
ordinary BatchNorm.

If an ungrouped loader also carries sample weights, `*Attn` chains receive
`(x, w)` because `masked_input(::MaskedModel, x, w) = (x, w)`. Zero weights
are then treated as padded keys. Keep sample weights strictly positive, or
use grouped `w` only as a 0/1 pad flag (as the grouped loader does).
