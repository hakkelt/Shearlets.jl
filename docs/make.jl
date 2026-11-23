using Documenter, Shearlets, MultiscaleFilterBanks, MultiscaleTransformsBase
using Documenter: Remotes

DocMeta.setdocmeta!(Shearlets, :DocTestSetup, :(using Shearlets, Random); recursive = true)
DocMeta.setdocmeta!(MultiscaleFilterBanks, :DocTestSetup, :(using MultiscaleFilterBanks, Random); recursive = true)

makedocs(;
    # MultiscaleFilterBanks and MultiscaleTransformsBase are listed because the
    # API page includes docstrings of names Shearlets re-exports; their other
    # names are documented in their own manuals, so `checkdocs` skips them.
    modules = [Shearlets, MultiscaleFilterBanks, MultiscaleTransformsBase],
    checkdocs_ignored_modules = [MultiscaleFilterBanks, MultiscaleTransformsBase],
    sitename = "Shearlets.jl",
    repo = Remotes.GitHub("hakkelt", "Shearlets.jl"),
    remotes = Dict(
        pkgdir(MultiscaleFilterBanks) => Remotes.GitHub("hakkelt", "MultiscaleFilterBanks.jl"),
        pkgdir(MultiscaleTransformsBase) => Remotes.GitHub("hakkelt", "MultiscaleTransformsBase.jl"),
    ),
    authors = "Tamás Hakkel",
    format = Documenter.HTML(; canonical = "https://hakkelt.github.io/Shearlets.jl"),
    pages = [
        "Home" => "index.md",
        "How-to guides" => [
            "Common tasks" => "howto.md",
            "Inspect responses and choose parameters" => "howto_parameters.md",
            "Examples" => "examples.md",
        ],
        "Explanation" => "theory.md",
        "Reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/hakkelt/Shearlets.jl", devbranch = "master")
