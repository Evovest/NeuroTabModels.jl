#!/usr/bin/env julia

# Render the reports and copy them into Documenter:
#
#   julia --project=quarto quarto/run.jl                    # every report
#   julia --project=quarto quarto/run.jl boston california  # those benchmarks, then the summary
#
# Set `BENCHMARK_DEVICE=gpu` to train the benchmarks on GPU; this needs CUDA and cuDNN added
# to the quarto environment.
#
# Reports are rendered in place and then moved: with `--output-dir`, Quarto runs Julia
# through Jupyter instead of its own Julia engine. The qmds use `format: gfm` with
# `variant: -raw_html` so figures are `![](stem_files/...)`, which DocumenterVitepress keeps
# as images. `benchmark.qmd` is rendered once per dataset, from a copy named
# `benchmark-<dataset>.qmd`, so each dataset gets its own page and figures.

include(joinpath(@__DIR__, "benchmarks", "benchmarks.jl"))

dest = joinpath(@__DIR__, "..", "docs", "src", "quarto")

function render(qmd; params...)
    args = reduce(vcat, ["-P", "$k:$v"] for (k, v) in params; init=String[])
    run(`quarto render $qmd $args`)
    stem = first(splitext(qmd))
    mv("$stem.md", joinpath(dest, "$stem.md"); force=true)
    isdir("$(stem)_files") && mv("$(stem)_files", joinpath(dest, "$(stem)_files"); force=true)
end

datasets = isempty(ARGS) ? sort(collect(keys(DATASETS))) : ARGS
for d in datasets
    haskey(DATASETS, d) || error("Unknown dataset `$d`. Datasets: $(join(sort(collect(keys(DATASETS))), ", ")).")
end

cd(@__DIR__) do
    mkpath(dest)
    isempty(ARGS) && render("embeddings-design.qmd")
    for d in datasets
        qmd = "benchmark-$d.qmd"
        cp("benchmark.qmd", qmd; force=true)
        try
            render(qmd; dataset=d, device=get(ENV, "BENCHMARK_DEVICE", "cpu"))
        finally
            rm(qmd; force=true)
        end
    end
    render("benchmarks.qmd")
end
