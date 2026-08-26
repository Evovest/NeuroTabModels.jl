#!/usr/bin/env julia

# Render named reports into quarto/build/, then copy into Documenter.
# Quarto requires --output-dir to be a subdirectory of the project (quarto/).
# The qmd uses `format: gfm` with `variant: -raw_html` so figures are
# `![](stem_files/...)`, which DocumenterVitepress keeps as images.
reports = [
    "embeddings-design.qmd",
]
build_dir = joinpath(@__DIR__, "build")
dest = joinpath(@__DIR__, "..", "docs", "src", "quarto")

cd(@__DIR__) do
    mkpath(build_dir)
    mkpath(dest)
    for report in reports
        run(`quarto render $report --output-dir $build_dir`)
        stem = first(splitext(report))
        cp(joinpath(build_dir, "$stem.md"), joinpath(dest, "$stem.md"); force=true)
        files = "$(stem)_files"
        src_files = joinpath(build_dir, files)
        if isdir(src_files)
            dst_files = joinpath(dest, files)
            isdir(dst_files) && rm(dst_files; recursive=true)
            cp(src_files, dst_files)
        end
    end
end
