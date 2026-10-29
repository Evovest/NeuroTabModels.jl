# pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Documenter
using DocumenterVitepress
using NeuroTabModels

# one page per rendered benchmark report, see quarto/benchmarks/run.jl
benchmark_pages = [
    titlecase(f[7:(end - 3)]) => "quarto/$f" for f in sort(readdir(joinpath(@__DIR__, "src", "quarto"))) if
    startswith(f, "bench-") && endswith(f, ".md") && f != "bench-summary.md"
]

pages = [
    "Quick start" => "quick-start.md",
    "API" => ["Training" => "API.md", "Losses" => "losses.md"],
    "Embeddings" => ["Design" => "quarto/embeddings-design.md", "API" => "embeddings.md"],
    "Models" => [
        "Interface" => "models/models.md",
        "MLP" => "models/mlp.md",
        "ResNet" => "models/resnet.md",
        "NeuroTrees" => ["design" => "design.md", "API" => "models/neurotrees.md"],
        "TabM" => "models/tabM.md",
        "ModernNCA" => ["design" => "models/modernnca-design.md", "API" => "models/modernnca.md"],
    ],
    "Layers" => "models/layers.md",
    "Padding and masks" => "models/padding-mask.md",
    "Tutorials" => [
        "Regression - Boston" => "tutorials/regression-boston.md",
        "Logistic - Titanic" => "tutorials/logistic-titanic.md",
        "Classification - IRIS" => "tutorials/classification-iris.md",
    ],
]
isfile(joinpath(@__DIR__, "src", "quarto", "bench-summary.md")) &&
    push!(pages, "Benchmarks" => ["Summary" => "quarto/bench-summary.md", benchmark_pages...])

makedocs(;
    sitename="NeuroTabModels",
    authors="Evovest and contributors.",
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/Evovest/NeuroTabModels.jl", # this must be the full URL!
        devbranch="main",
        devurl="dev",
    ),
    modules=[NeuroTabModels],
    warnonly=true,
    checkdocs=:all,
    pages=pages,
)

DocumenterVitepress.deploydocs(;
    repo="github.com/Evovest/NeuroTabModels.jl", # this must be the full URL!
    target=joinpath(@__DIR__, "build"),
    branch="gh-pages",
    devbranch="main",
    push_preview=true,
)
