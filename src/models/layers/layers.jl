module Layers

export MaskedBatchNorm, CarryMask, MaskSkip
export GroupedDense, rsqrt_uniform_grouped, glorot_uniform_grouped
export AttnResidual

using Lux
using Lux: StatefulLuxLayer, zeros32
using LuxCore
using LuxLib: batched_matmul
using NNlib: dot_product_attention
using Random: AbstractRNG

include("maskednorm.jl")
include("attn.jl")
include("groupeddense.jl")

end
