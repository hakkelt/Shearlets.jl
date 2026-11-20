# Wire the unregistered sibling checkouts of MultiscaleFilterBanks.jl and
# MultiscaleTransformsBase.jl into every environment of this workspace.
#
#     julia --project=. dev/bootstrap.jl
#
# `[sources]` in Project.toml is the declared mechanism and covers Julia ≥ 1.11.
# This package also supports 1.10, which does not read `[sources]`, so there the
# path dependencies have to be recorded in the gitignored Manifest.toml instead.
# Run this once per fresh checkout; re-running it is harmless.
using Pkg

const SIBLINGS = [normpath(joinpath(@__DIR__, "..", "..", name)) for name in ("MultiscaleTransformsBase", "MultiscaleFilterBanks")]
for dir in SIBLINGS
    isdir(dir) || error("sibling checkout not found at $dir")
end

const ROOT = normpath(joinpath(@__DIR__, ".."))

for env in (ROOT, joinpath(ROOT, "test"), joinpath(ROOT, "docs"), joinpath(ROOT, "benchmark"))
    isfile(joinpath(env, "Project.toml")) || continue
    @info "developing MultiscaleTransformsBase and MultiscaleFilterBanks into" env
    Pkg.activate(env)
    Pkg.develop([PackageSpec(path = dir) for dir in SIBLINGS])
end

Pkg.activate(ROOT)
