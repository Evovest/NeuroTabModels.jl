module NeuroTabModelsZygoteExt

using Lux: AutoZygote
using NeuroTabModels
using Zygote

import NeuroTabModels.Fit: get_ad_backend

get_ad_backend(::Val{:zygote}) = AutoZygote()

end
