using NeuroTabModels
using NeuroTabModels.Models.NeuroTreeModels

# node-weights => leaf-weights
# [N, T, B] -> [L, T, B]
depth = 3
ntrees = 4
ns = 2^depth - 1
ls = node_size + 1
bs = 128

nw = rand(ns, ntrees, bs)
