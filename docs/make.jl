using DDMFaultSlip
using Documenter

# Headless plotting with GR
ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(DDMFaultSlip, :DocTestSetup, :(using DDMFaultSlip); recursive=true)

makedocs(;
    modules=[DDMFaultSlip],
    authors="Antoine Jacquey <antoine.jacquey@polymtl.ca> and contributors",
    repo=Documenter.Remotes.GitHub("ajacquey", "DDMFaultSlip.jl"),
    sitename="DDMFaultSlip.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ajacquey.github.io/DDMFaultSlip.jl",
        edit_link="main",
        assets=String[],
    ),
    checkdocs=:exports,
    pages=[
        "Home" => "index.md",
        "Background" => "background.md",
        "User guide" => "guide.md",
        "Tutorials" => [
            "Fluid-induced aseismic slip" => "tutorials/fluid_injection.md",
        ],
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/ajacquey/DDMFaultSlip.jl",
    devbranch="main",
)
