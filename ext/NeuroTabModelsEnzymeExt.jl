module NeuroTabModelsEnzymeExt

using Enzyme
using Lux: AutoEnzyme
using NeuroTabModels

import NeuroTabModels.Fit: get_ad_backend

get_ad_backend(::Val{:enzyme}) = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

end
