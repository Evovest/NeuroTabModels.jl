using Random
using Lux
using DataFrames

rng = Random.Xoshiro(123)
batch = 16
nfeats = 3
embed_lvls = 7
embed_size = 9

struct NumCatEmbeddings{L} <: AbstractLuxWrapperLayer{:layer}
    layer::L
end

function NumCatEmbeddings(; num_size=5, cat_size=0, embed_lvls=7, embed_size=9)

    # numerical embeddings
    nums = BatchNorm(num_size)

    # categorical embeddings
    cats = []
    for i in 1:cat_size
        push!(cats, Symbol("embed$i") => Embedding(embed_lvls => embed_size))
    end
    m = Parallel(vcat; name="preproc", nums, cats...)
    return NumCatEmbeddings(m)
end

m = NumCatEmbeddings(; num_size=nfeats, cat_size=2)
ps, st = LuxCore.setup(rng, m)

x = randn(rng, Float32, nfeats, batch)
c1 = rand(rng, 1:embed_lvls, batch)
c2 = rand(rng, 1:embed_lvls, batch)

out, _st = m((x, c1, c2), ps, Lux.testmode(st))
