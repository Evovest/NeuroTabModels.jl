using NeuroTabModels

arch = NeuroTabModels.NeuroTreeConfig(;
    tree_type=:binary,
    depth=4,
)
learner = NeuroTabRegressor(
    arch;
    nrounds=10,
)
isnothing(learner.embedding_config)

embedding_config = nothing
learner = NeuroTabRegressor(
    arch;
    embedding_config,
    nrounds=10,
)
isnothing(learner.embedding_config)

embedding_config = NeuroTabModels.EmbeddingLayer(num=NeuroTabModels.BatchNormEmbeddings())
learner = NeuroTabRegressor(
    arch;
    embedding_config,
    nrounds=10,
)
typeof(learner.embedding_config) <: NeuroTabModels.EmbeddingLayer
typeof(learner.embedding_config.num) <: NeuroTabModels.BatchNormEmbeddings
isnothing(learner.embedding_config.temp)



embedding_dict = Dict(:num_type => :batchnorm)
embedding_config = NeuroTabModels.EmbeddingLayer()
learner = NeuroTabRegressor(
    arch;
    embedding_config,
    nrounds=10,
)
typeof(learner.embedding_config) <: NeuroTabModels.EmbeddingLayer
typeof(learner.embedding_config.num) <: NeuroTabModels.BatchNormEmbeddings
isnothing(learner.embedding_config.temp)
