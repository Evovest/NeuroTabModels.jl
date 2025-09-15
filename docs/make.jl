# push!(LOAD_PATH, "../src/")
using Documenter
using DocumenterVitepress
using NeuroTabModels

ages = [
    "Quick start" => "quick-start.md",
    "Design" => "design.md",
    "Models" => "models.md",
    "API" => "API.md",
    "Tutorials" => [
        "Regression - Boston" => "tutorials/regression-boston.md",
        "Logistic - Titanic" => "tutorials/logistic-titanic.md",
        "Classification - IRIS" => "tutorials/classification-iris.md",
    ]
]

makedocs(;
    sitename="NeuroTabModels",
    authors="Evovest and contributors.",
    format=DocumenterVitepress.MarkdownVitepress(
        repo="github.com/Evovest/NeuroTabModels.jl", # this must be the full URL!
        devbranch="main",
        devurl="dev";
    ),
    modules=[NeuroTabModels],
    warnonly=true,
    checkdocs=:all,
    pages=pages,
)

deploydocs(;
    repo="github.com/Evovest/NeuroTabModels.jl", # this must be the full URL!
    target="build", # this is where Vitepress stores its output
    branch="gh-pages",
    devbranch="main",
    push_preview=true
)
