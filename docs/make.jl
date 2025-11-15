using Pkg
Pkg.activate("docs")
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()

using MRIRealign
using Documenter

DocMeta.setdocmeta!(MRIRealign, :DocTestSetup, :(using MRIRealign); recursive=true)

makedocs(;
    doctest=true,
    doctestfilters = [r"\s*-?(\d+)\.(\d{4})\d*\s*"], # Ignore any digit after the 4th digit after a decimal, throughout the docs
    modules=[MRIRealign],
    authors="Jakob Asslaender <jakob.asslaender@nyumc.org> and contributors",
    repo = Documenter.Remotes.GitHub("MagneticResonanceImaging", "MRIRealign.jl"),
    sitename="MRIRealign.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MagneticResonanceImaging.github.io/MRIRealign.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

# Set dark theme as default independent of the OS's settings
run(`sed -i'.old' 's/var darkPreference = false/var darkPreference = true/g' docs/build/assets/themeswap.js`)

deploydocs(;
    repo="github.com/MagneticResonanceImaging/MRIRealign.jl",
    push_preview=true,
)
