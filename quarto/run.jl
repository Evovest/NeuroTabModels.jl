#!/usr/bin/env julia

report = "embeddings.qmd"
output_name = "embeddings-design.md"
output_dir = "../docs/src/quarto"

cd(@__DIR__) do
    run(`quarto render $report --output $output_name --output-dir $output_dir`)
end
