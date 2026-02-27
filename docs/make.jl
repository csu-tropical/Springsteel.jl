using Springsteel
using Documenter

DocMeta.setdocmeta!(Springsteel, :DocTestSetup, :(using Springsteel); recursive=true)

makedocs(;
    modules=[Springsteel, Springsteel.CubicBSpline, Springsteel.Fourier, Springsteel.Chebyshev],
    authors="Michael Bell <mmbell@colostate.edu> and contributors",
    sitename="Springsteel.jl",
    format=Documenter.HTML(;
        canonical="https://mmbell.github.io/Springsteel.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "CubicBSpline" => "cubicbspline.md",
        "Fourier" => "fourier.md",
        "Chebyshev" => "chebyshev.md",
        "Testing Guide" => "testing_guide.md",
        "Documentation Guide" => "documentation_guide.md",
    ],
)

deploydocs(;
    repo="github.com/mmbell/Springsteel.jl",
    devbranch="main",
)
