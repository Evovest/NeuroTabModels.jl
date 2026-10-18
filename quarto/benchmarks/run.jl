#!/usr/bin/env julia

# Render the benchmark report of each dataset given, then the summary, into Documenter:
#
#   julia --project=quarto quarto/benchmarks/run.jl boston california
#   julia --project=quarto quarto/benchmarks/run.jl --device=cpu --backend=zygote boston
#
# The datasets are the keys of `DATASETS` in `benchmarks.jl`. `--device` is `gpu` (default)
# or `cpu`, `--backend` is `reactant` (default) or `zygote`. `--device=gpu` with Zygote needs
# CUDA and cuDNN added to the quarto environment.
#
# `bench-template.qmd` is rendered once per dataset as `bench-<dataset>.md`. Quarto still
# writes its figures to `bench-template_files`, so they are moved to `bench-<dataset>_files`
# and the links rewritten. `_quarto.yml` keeps Quarto on its Julia engine: without it,
# `--output-dir` runs Julia through Jupyter.

dest = joinpath(@__DIR__, "..", "..", "docs", "src", "quarto")

options = Dict("device" => "gpu", "backend" => "reactant")
datasets = String[]
for a in ARGS
    m = match(r"^--(device|backend)=(\w+)$", a)
    isnothing(m) ? push!(datasets, a) : (options[m[1]] = m[2])
end
isempty(datasets) && error("Give the datasets to render, for example `run.jl boston california`.")

cd(@__DIR__) do
    for d in datasets
        stem = "bench-$d"
        run(`quarto render bench-template.qmd -P dataset:$d -P device:$(options["device"]) -P backend:$(options["backend"]) --output $stem.md --output-dir $dest`)
        md = joinpath(dest, "$stem.md")
        write(md, replace(read(md, String), "bench-template_files" => "$(stem)_files"))
        mv(joinpath(dest, "bench-template_files"), joinpath(dest, "$(stem)_files"); force=true)
        # a new worker per dataset, so none holds on to GPU memory while the next one trains
        run(`quarto call engine julia close $(abspath("bench-template.qmd"))`)
    end
    run(`quarto render bench-summary.qmd --output-dir $dest`)
end
