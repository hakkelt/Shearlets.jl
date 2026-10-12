using Documenter, Shearlets, MultiscaleFilterBanks
using Documenter: Remotes

DocMeta.setdocmeta!(Shearlets, :DocTestSetup, :(using Shearlets, Random); recursive = true)
DocMeta.setdocmeta!(MultiscaleFilterBanks, :DocTestSetup, :(using MultiscaleFilterBanks, Random); recursive = true)

makedocs(;
    # MultiscaleFilterBanks is listed because Shearlets re-exports its public
    # names (`FilterPair`, `CDF97`, the threading policies).
    modules = [Shearlets, MultiscaleFilterBanks],
    sitename = "Shearlets.jl",
    repo = Remotes.GitHub("hakkelt", "Shearlets.jl"),
    remotes = Dict(pkgdir(MultiscaleFilterBanks) => Remotes.GitHub("hakkelt", "MultiscaleFilterBanks.jl")),
    authors = "Tamás Hakkel",
    format = Documenter.HTML(; canonical = "https://hakkelt.github.io/Shearlets.jl"),
    pages = [
        "Home" => "index.md",
        "Shearlet systems" => "systems.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/hakkelt/Shearlets.jl", devbranch = "master")
