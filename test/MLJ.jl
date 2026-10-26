logit(x) = log(x / (1 - x))
logit(x::AbstractVector) = logit.(x)
sigmoid(x) = 1 / (1 + exp(-x))
sigmoid(x::AbstractVector) = sigmoid.(x)

@testset "generic interface tests" begin
    @testset "NeuroTabRegressor" begin
        failures, summary = MLJTestInterface.test(
            [NeuroTabRegressor],
            MLJTestInterface.make_regression()...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=true, # set to true to debug
        )
        @test isempty(failures)
    end
    @testset "NeuroTabClassifier" begin
        failures, summary = MLJTestInterface.test(
            [NeuroTabClassifier],
            MLJTestInterface.make_binary()...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=true, # set to true to debug
        )
        @test isempty(failures)

        failures, summary = MLJTestInterface.test(
            [NeuroTabClassifier],
            MLJTestInterface.make_multiclass()...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=true, # set to true to debug
        )
        @test isempty(failures)
    end
end

##################################################
### Regression
##################################################
@testset "MLJ - regression" begin
    features = rand(1_000) .* 5 .- 2
    X = reshape(features, (size(features)[1], 1))
    Y = sin.(features) .* 0.5 .+ 0.5
    Y = logit(Y) + randn(size(Y))
    Y = sigmoid(Y)
    y = Y
    X = MLJBase.table(X)

    tree_model = NeuroTabRegressor(; arch_name="NeuroTreeConfig")

    mach = machine(tree_model, X, y)
    train, test = partition(eachindex(y), 0.7, shuffle=true) # 70:30 split
    fit!(mach, rows=train, verbosity=1)

    mach.model.nrounds += 10
    fit!(mach, rows=train, verbosity=1)
    _report = report(mach)

    # predict on train data
    pred_train = predict(mach, selectrows(X, train))
    mean(abs.(pred_train - selectrows(Y, train)))

    # predict on test data
    pred_test = predict(mach, selectrows(X, test))
    mean(abs.(pred_test - selectrows(Y, test)))

    @test MLJBase.iteration_parameter(NeuroTabRegressor) == :nrounds
end

@testset "MLJ - update refits when a hyperparameter other than nrounds changes" begin
    X, y = make_regression(500, 3)
    mach = machine(NeuroTabRegressor(; nrounds=2, lr=1e-2, seed=1), X, y)
    fit!(mach, verbosity=0)
    fr = mach.fitresult

    # more rounds only: training goes on from the fitted parameters
    mach.model.nrounds = 4
    fit!(mach, verbosity=0)
    @test mach.fitresult === fr
    @test mach.fitresult.info[:nrounds] == 4

    # any other change: a fresh fit, the same as fitting the changed model from scratch
    mach.model.lr = 1e-3
    fit!(mach, verbosity=0)
    @test mach.fitresult !== fr
    fresh = machine(NeuroTabRegressor(; nrounds=4, lr=1e-3, seed=1), X, y)
    fit!(fresh, verbosity=0)
    @test predict(mach, X) ≈ predict(fresh, X)
end

@testset "MLJ - rowtables - NeuroTabRegressor" begin
    X, y = make_regression(1000, 5)
    X = Tables.rowtable(X)
    booster = NeuroTabRegressor()
    # smoke tests:
    mach = machine(booster, X, y) |> fit!
    fit!(mach)
    report(mach)
    predict(mach, X)
end

@testset "MLJ - named tuples - NeuroTabRegressor" begin
    X, y = (x1=rand(100), x2=rand(100)), rand(100)
    booster = NeuroTabRegressor()
    # smoke tests:
    mach = machine(booster, X, y) |> fit!
    fit!(mach)
    report(mach)
    predict(mach, X)
end

@testset "MLJ - classification" begin
    X, y = @load_crabs

    tree_model = NeuroTabClassifier(; arch_name="NeuroTreeConfig", lr=0.1, nrounds=20, batchsize=64)

    # @load EvoTreeRegressor
    mach = machine(tree_model, X, y)
    train, test = partition(eachindex(y), 0.7, shuffle=true) # 70:30 split
    fit!(mach, rows=train, verbosity=1)

    mach.model.nrounds += 50
    fit!(mach, rows=train, verbosity=1)

    pred_train = predict(mach, selectrows(X, train))
    pred_train_mode = predict_mode(mach, selectrows(X, train))
    sum(pred_train_mode .== y[train]) / length(y[train])

    pred_test = predict(mach, selectrows(X, test))
    pred_test_mode = predict_mode(mach, selectrows(X, test))
    pred_test_mode = predict_mode(mach, selectrows(X, test))
    sum(pred_test_mode .== y[test]) / length(y[test])
end

@testset "MLJ - support for ordered factor predictions" begin
    X = (; x=rand(10))
    y = coerce(rand("ab", 10), OrderedFactor)
    model = NeuroTabClassifier(; nrounds=10)
    mach = machine(model, X, y) |> fit!
    yhat = predict(mach, X)
    @assert isordered(yhat)
end
