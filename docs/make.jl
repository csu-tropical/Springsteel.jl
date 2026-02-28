using Springsteel
using Documenter

DocMeta.setdocmeta!(Springsteel, :DocTestSetup, :(using Springsteel); recursive=true)

makedocs(;
    modules=[Springsteel, Springsteel.CubicBSpline, Springsteel.Fourier, Springsteel.Chebyshev],
    authors="Michael Bell <mmbell@colostate.edu> and contributors",
    sitename="Springsteel.jl",
    warnonly=[:missing_docs, :cross_references],
    format=Documenter.HTML(;
        canonical="https://mmbell.github.io/Springsteel.jl",
        edit_link="main",
        assets=String[],
        prettyurls=get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "SpringsteelGrid" => "springsteel_grid.md",
        "CubicBSpline" => "cubicbspline.md",
        "Fourier" => "fourier.md",
        "Chebyshev" => "chebyshev.md",
        "Testing Guide" => "testing_guide.md",
        "Documentation Guide" => "documentation_guide.md",
        "Developer Notes" => "developer_notes.md",
    ],
)

deploydocs(;
    repo="github.com/mmbell/Springsteel.jl",
    devbranch="main",
)
