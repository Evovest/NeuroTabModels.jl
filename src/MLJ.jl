module MLJ

using Tables
using DataFrames
import ..Learners: NeuroTabRegressor, NeuroTabClassifier, LearnerTypes
import ..Fit: init, fit_iter!, _sync_params_to_model!
import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema

export fit, update, predict

function fit(model::LearnerTypes, verbosity::Int, A, y, w=nothing)
    Tables.istable(A) ? dtrain = DataFrame(A) : error("`A` must be a Table")
    feature_names = string.(collect(Tables.schema(dtrain).names))
    @assert "_target" ∉ feature_names
    dtrain._target = y
    target_name = "_target"

    if !isnothing(w)
        @assert "_weight" ∉ feature_names
        dtrain._weight = w
        weight_name = "_weight"
    else
        weight_name = nothing
    end
    offset_name = nothing

    fitresult, cache = init(model, dtrain; feature_names, target_name, weight_name, offset_name)

    while fitresult.info[:nrounds] < model.nrounds
        fit_iter!(fitresult, cache)
    end

    _sync_params_to_model!(fitresult, cache)
    cache[:model] = _snapshot(model)
    report = (features=fitresult.info[:feature_names],)
    return fitresult, cache, report
end

# The learner as fitted, kept so `update` can tell what MLJ changed since. Its field values are
# immutable, so sharing them is enough; a deep copy would give the vectors of a temporal embedding
# config new identities, and an unchanged model would then no longer compare equal.
_snapshot(model::T) where {T<:LearnerTypes} = T((getfield(model, f) for f in fieldnames(T))...)

# Training can only go on from the fitted parameters, so continuing is valid when the model
# differs from the fitted one in `nrounds` alone and asks for at least as many rounds. Any other
# change, such as `lr`, `batchsize` or the architecture, needs a fresh fit.
function okay_to_continue(model, fitresult, cache)
    return MMI.is_same_except(model, cache[:model], :nrounds) &&
           model.nrounds >= fitresult.info[:nrounds]
end

MMI.iteration_parameter(::Type{<:LearnerTypes}) = :nrounds

function update(model::LearnerTypes, verbosity::Integer, fitresult, cache, A, y, w=nothing)
    if okay_to_continue(model, fitresult, cache)
        while fitresult.info[:nrounds] < model.nrounds
            fit_iter!(fitresult, cache)
        end
        _sync_params_to_model!(fitresult, cache)
        cache[:model] = _snapshot(model)
        report = (features=fitresult.info[:feature_names],)
    else
        fitresult, cache, report = fit(model, verbosity, A, y, w)
    end
    return fitresult, cache, report
end

function predict(::NeuroTabRegressor, fitresult, A)
    Tables.istable(A) ? df = DataFrame(A) : error("`A` must be a Table")
    pred = fitresult(df)
    return pred
end

function predict(::NeuroTabClassifier, fitresult, A)
    Tables.istable(A) ? df = DataFrame(A) : error("`A` must be a Table")
    pred = fitresult(df)
    return MMI.UnivariateFinite(fitresult.info[:target_levels], pred)
end

MMI.metadata_pkg.(
    (NeuroTabRegressor, NeuroTabClassifier);
    name="NeuroTabModels",
    uuid="1db4e0a5-a364-4b0c-897c-2bd5a4a3a1f2",
    url="https://github.com/Evovest/NeuroTabModels.jl",
    julia=true,
    license="Apache",
    is_wrapper=false,
)

MMI.metadata_model(
    NeuroTabRegressor;
    input_scitype=MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor),
    target_scitype=AbstractVector{<:MMI.Continuous},
    weights=true,
    path="NeuroTabModels.NeuroTabRegressor",
)

MMI.metadata_model(
    NeuroTabClassifier;
    input_scitype=MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor),
    target_scitype=AbstractVector{<:MMI.Finite},
    weights=true,
    path="NeuroTabModels.NeuroTabClassifier",
)

end