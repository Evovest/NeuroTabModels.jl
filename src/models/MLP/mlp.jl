module MLPs

import Flux
import Flux: @functor, trainmode!, gradient, Chain, DataLoader, cpu, gpu
import Flux: logσ, logsoftmax, softmax, softmax!, relu, sigmoid, sigmoid_fast, hardsigmoid, tanh, tanh_fast, hardtanh, softplus, onecold, onehotbatch
import Flux: BatchNorm, Dense, MultiHeadAttention, Parallel, SkipConnection

import ..Models: get_loss_type, GaussianMLE
import ..Models: get_model_chain, ModelType

function get_model_chain(::Type{ModelType{:MLP}}, config; nfeats, outsize, kwargs...)

    L = get_loss_type(config.loss)
    hsize = 64

    if L <: GaussianMLE && config.MLE_tree_split
        chain = Chain(
            BatchNorm(nfeats),
            Parallel(
                vcat,
                Chain(
                    BatchNorm(nfeats),
                    Dense(nfeats => hsize, relu),
                    BatchNorm(hsize),
                    Dense(hsize => outsize)
                ),
                Chain(
                    BatchNorm(nfeats),
                    Dense(nfeats => hsize, relu),
                    BatchNorm(hsize),
                    Dense(hsize => outsize)
                )
            )
        )
    else
        outsize = L <: GaussianMLE ? 2 * outsize : outsize
        # chain = Chain(
        #     BatchNorm(nfeats),
        #     Dense(nfeats => hsize),
        #     BatchNorm(hsize, relu),
        #     Dense(hsize => hsize),
        #     BatchNorm(hsize, relu),
        #     Dense(hsize => outsize)
        # )
        chain = Chain(
            BatchNorm(nfeats),
            Dense(nfeats => hsize),
            BatchNorm(hsize, relu),
            # SkipConnection(Chain(
            #     Dense(hsize => hsize),
            #     BatchNorm(hsize, relu)),
            #     +
            # ),
            # Dense(2 * hsize => hsize),
            SkipConnection(Chain(
                    Dense(hsize => hsize),
                    BatchNorm(hsize, relu)),
                vcat
            ),
            # x -> relu.(x),
            Dense(2 * hsize => outsize)
        )
    end

    return chain
end

end