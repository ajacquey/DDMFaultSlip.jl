using DDMFaultSlip
using Documenter

DocMeta.setdocmeta!(DDMFaultSlip, :DocTestSetup, :(using DDMFaultSlip); recursive=true)

makedocs(;
    modules=[DDMFaultSlip],
    authors="Antoine Jacquey <antoine.jacquey@tufts.edu> and contributors",
    repo="https://github.com/ajacquey/DDMFaultSlip.jl/blob/{commit}{path}#{line}",
    sitename="DDMFaultSlip.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ajacquey.github.io/DDMFaultSlip.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ajacquey/DDMFaultSlip.jl",
    devbranch="main",
)
