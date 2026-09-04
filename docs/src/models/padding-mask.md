# Grouped padding and masks

Set mixers (`MLPAttn`, `NeuroTreeAttn`) treat the current batch — or one
group, when `group_name` is set — as an **unordered set of tokens**.
Grouped loaders pad those sets to a common buffer width. This page is
the design for that padding: where `w` comes from, why embeddings never
see it, and how the core consumes it.

## Why there is a pad at all

A grouped train/eval/infer loader takes `groupby(df, group_name)` and
writes **one padded tensor per group**. The buffer width is the size of
the largest group, so every group is a dense `(features, buffer)`
matrix:

```
one group, 3 real rows, buffer = 5

        col:   1    2    3    4    5
               ├──── real ────┤  pads

   x  =  [  x₁   x₂   x₃   0    0  ]     (nfeats × buffer)
   y  =  [  y₁   y₂   y₃   0    0  ]
   w  =  [  1    1    1    0    0  ]     1 = real row, 0 = pad slot
```

Train/eval use `w` as that 0/1 flag (shape `(1, 1, buffer)`). Infer uses
a boolean `mask` of length `buffer` and, after the forward, keeps only
`pred[:, mask]`.

Pads exist so the group is a rectangular array. They are **not**
observations. Anything that reduces or mixes across the last dimension
must ignore them. Anything that maps **one column independently** can
embed a pad column without contaminating real columns.

## Two jobs for `w`

`w` is reused for two different things, at two different places:

| Job | Who uses it | What happens to pads |
| --- | --- | --- |
| **Loss / metric weight** | `_reduce(loss, w)` and eval | pad slots contribute `0` to the scalar loss |
| **Set-mixer mask** | `MLPAttn` / `NeuroTreeAttn` | pads are excluded from attention keys and from `MaskedBatchNorm` stats |

A non-mixer (`ResNet`, `MLP`, …) only needs the first job. Its forward
sees `x` only; pad rows still produce a prediction, but that prediction
is multiplied by `w = 0` in the loss. That is why grouped training works
for those models **without** threading a mask through the net.

A mixer also needs the second job. Attention is a reduction over keys:
a pad token with value `0` would still be a key every query can attend
to. Encoder BatchNorm is a reduction over the batch: pad zeros would
enter `μ` / `σ` and running stats. Those layers must see a valid-token
flag.

## Why embeddings do not take `w`

Embeddings are **per-observation maps**. A linear / periodic /
piecewise-linear / identity / layer-norm embedding applied to column
`j` does not read column `k`. Computing `embed(x_pad)` cannot change
`embed(x_real)`.

So the mask is not an embedding concern. Forcing every embedding layer
to accept `(x, w)` would mean wrapping `Dense`-like column maps in
tuple plumbing for a flag they do not use.

The split is implemented once, at the **embed / core boundary**:

```
input is either x  or  (x, w)

                    (x, w)
                       │
          ┌────────────┴────────────┐
          │      MaskedModel        │
          │                         │
          │   z = embed(x)          │  ← embeddings: x only
          │                         │
          │   y = core((z, w))      │  ← mixer / MaskedBN: (z, w)
          └────────────┬────────────┘
                       │
                       y
```

`fit` chooses that wrapper from the core, not from the loader:

```julia
chain = if uses_batch_mask(core_chain)
    MaskedModel(embed_chain, core_chain)   # mixers
else
    Chain(embed_chain, core_chain)          # everyone else
end
```

`Chain(embed, core)` would pass `(x, w)` into `embed`, which does not
implement that signature. `MaskedModel` peels the tuple: embeddings stay
ordinary Lux layers; only the core is mask-aware.

Call sites do not special-case embeddings either. `masked_input`
decides whether `w` is forwarded **into the assembled chain**:

```
default:   masked_input(model, x, w) = x          # Chain(...)
mixer:     masked_input(model, x, w) = (x, w)    # MaskedModel
```

Loss, eval, and infer all go through that hook. The loader always
produces `(x, y, w)` for grouped data; only mask-aware chains unpack `w`
as a model input.

### What embeddings still do with pads

Pad columns are still **forwarded**. After `embed(x)` you have tokens
for real rows **and** for zeros in the buffer. That is intentional:

- the core keeps a rectangular `(hidden, buffer)` stream;
- attention / `MaskedBatchNorm` then drop pads from reductions;
- the head still emits a prediction per slot, and the loss zeros the pad
  slots.

Embeddings are allowed to produce garbage on pad columns. Those tokens
must not leak into real tokens downstream.

`BatchNormEmbeddings` is the exception: it **does** reduce across the
batch, and today it does not see `w`. Prefer identity / linear / LN
embeddings (or the mixer’s own `MaskedBatchNorm`) when training a set
mixer on grouped data.

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

## Inside the mixer

`MLPAttn` converts `w` to a boolean `valid` once, then uses the **same
flags** in two shapes.

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
   │   rectangular key-padding mask (not causal):
   │   padded keys are zeroed for every query
   │
   ▼
head(z)        Dense on every column, including pads
```

`CarryMask` / `MaskSkip` exist so a `Chain` of ordinary layers can thread
`(features, mask)` without each `Dense` knowing about padding.
`MaskedBatchNorm` is the layer that actually **reads** the flag.

`NeuroTreeAttn` is the same split with a tree encoder: trees are
per-observation (`CarryMask` around `NeuroTree` + flatten), then
`MaskedBatchNorm` uses the valid-token flag like MLPAttn. Attention
blocks receive a rectangular key-padding mask. The hidden width is
NeuroTree `k` (one ensemble of `ntrees` per channel, scalar leaf
preds), not the number of leaves.

Attention’s own batch dimension is always `1`. The “sequence” is the
group / minibatch. The mask is therefore a **key-padding** mask over
observations, not a feature mask and not a causal triangle.

## Ungrouped batches

With no `group_name`, there are no pad slots. Every column is a real
observation. Mixers still accept a plain `x` (the unmasked method).
`n_attn_layers=0` on `MLPAttn` is that encoder plus a linear head: a
ResNet ablation, including `MaskedBatchNorm` with `valid = nothing`,
which is ordinary BatchNorm.

If an ungrouped loader also carries sample weights, mixer chains receive
`(x, w)` because `masked_input(::MaskedModel, x, w) = (x, w)`. Zero
weights are then treated as padded keys. Keep sample weights strictly
positive, or use grouped `w` only as a 0/1 pad flag (as the grouped
loader does).
