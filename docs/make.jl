# push!(LOAD_PATH, "../src/")
using Documenter
using DocumenterVitepress
using NeuroTabModels

pages = [
    "Quick start" => "quick-start.md",
    "Design" => "design.md",
    "Models" => "models.md",
    "API" => "API.md",
    # "Tutorials" => [
    #     "Regression - Boston" => "tutorials-regression-boston.md",
    #     "Logistic - Titanic" => "tutorials-logistic-titanic.md",
    #     "Classification - IRIS" => "tutorials-classification-iris.md",
    # ]
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


# pages = [
#     "Quick start" => "quick-start.md",
#     "Design" => "design.md",
#     "Models" => "models.md",
#     "API" => "API.md",
#     "Tutorials" => [
#         "Regression - Boston" => "tutorials/regression-boston.md",
#         "Logistic - Titanic" => "tutorials/logistic-titanic.md",
#         "Classification - IRIS" => "tutorials/classification-iris.md",
#     ]
# ]
# makedocs(
#     sitename="NeuroTabModels",
#     authors="Jeremie Desgagne-Bouchard and contributors.",
#     format=Documenter.HTML(
#         sidebar_sitename=false,
#         edit_link="main",
#         assets=["assets/style.css"]
#     ),
#     modules=[NeuroTabModels],
#     pages=pages,
#     warnonly=true,
#     draft=false,
#     source="src",
#     build=joinpath(@__DIR__, "build")
# )

# deploydocs(
#     repo="github.com/Evovest/NeuroTabModels.jl.git",
#     target="build",
#     devbranch="main",
#     devurl="dev",
#     versions=["stable" => "v^", "v#.#", "dev" => "dev"],
# )
