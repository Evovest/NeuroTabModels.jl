# Layers

Reusable Lux layers shared by embeddings and backbones.

`CarryMask`, `MaskSkip`, `MaskedBatchNorm`, and `AttnResidual` take a mask
that marks which columns are real observations versus padding. Grouped
attention models use that so pad slots are ignored in attention and batch
norm. See [Grouped padding and masks](@ref).

```@autodocs
Modules = [NeuroTabModels.Models.Layers]
Private = false
Order = [:type, :function]
```
