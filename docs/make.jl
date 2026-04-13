using Springsteel
using Documenter

DocMeta.setdocmeta!(Springsteel, :DocTestSetup, :(using Springsteel); recursive=true)

makedocs(;
    modules=[Springsteel, Springsteel.CubicBSpline, Springsteel.Fourier, Springsteel.Chebyshev],
    authors="Michael Bell <mmbell@colostate.edu> and contributors",
    sitename="Springsteel.jl",
    warnonly=[:missing_docs, :cross_references, :docs_block],
    format=Documenter.HTML(;
        canonical="https://csu-tropical.github.io/Springsteel.jl",
        edit_link="main",
        assets=String[],
        prettyurls=get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "SpringsteelGrid" => "springsteel_grid.md",
        "Boundary Conditions" => "boundary_conditions.md",
        "Solver Framework" => "solver.md",
        "Multi-Patch Grids" => "multipatch.md",
        "Interpolation" => "interpolation.md",
        "Grid Relocation" => "relocation.md",
        "Spectral Filtering" => "filtering.md",
        "CubicBSpline" => "cubicbspline.md",
        "Fourier" => "fourier.md",
        "Chebyshev" => "chebyshev.md",
        "Contributing" => "contributing.md",
    ],
)

deploydocs(;
    repo="github.com/csu-tropical/Springsteel.jl",
    devbranch="development",
)
