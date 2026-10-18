#!/usr/bin/env julia

# Render the benchmark report of each dataset, then the summary, and copy them into Documenter:
#
#   julia --project=quarto quarto/benchmarks/run.jl                     # every dataset
#   julia --project=quarto quarto/benchmarks/run.jl boston california   # those datasets
#   julia --project=quarto quarto/benchmarks/run.jl --device=gpu --backend=reactant higgs
#
# `--device` is `cpu` (default) or `gpu`, `--backend` is `zygote` (default) or `reactant`.
# `--backend=reactant` needs Reactant, and `--device=gpu` with Zygote needs CUDA and cuDNN,
# added to the quarto environment.
#
# `bench-template.qmd` is rendered once per dataset as `bench-<dataset>.md`. Quarto still
# writes its figures to `bench-template_files`, so they are moved to `bench-<dataset>_files`
# and the links rewritten. Reports are rendered in place and then moved: with `--output-dir`,
# Quarto runs Julia through Jupyter instead of its own Julia engine.

include(joinpath(@__DIR__, "benchmarks.jl"))

dest = joinpath(@__DIR__, "..", "..", "docs", "src", "quarto")

options = Dict("device" => "cpu", "backend" => "zygote")
datasets = String[]
for a in ARGS
    m = match(r"^--(device|backend)=(\w+)$", a)
    isnothing(m) ? push!(datasets, a) : (options[m[1]] = m[2])
end
isempty(datasets) && append!(datasets, sort(collect(keys(DATASETS))))
for d in datasets
    haskey(DATASETS, d) || error("Unknown dataset `$d`. Datasets: $(join(sort(collect(keys(DATASETS))), ", ")).")
end

cd(@__DIR__) do
    mkpath(dest)
    for d in datasets
        stem = "bench-$d"
        run(`quarto render bench-template.qmd -P dataset:$d -P device:$(options["device"]) -P backend:$(options["backend"]) --output $stem.md`)
        write(joinpath(dest, "$stem.md"), replace(read("$stem.md", String), "bench-template_files" => "$(stem)_files"))
        rm("$stem.md")
        mv("bench-template_files", joinpath(dest, "$(stem)_files"); force=true)
    end
    run(`quarto render bench-summary.qmd`)
    mv("bench-summary.md", joinpath(dest, "bench-summary.md"); force=true)
end
