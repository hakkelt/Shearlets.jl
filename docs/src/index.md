# Shearlets.jl

Translation-invariant discrete shearlet transforms of images and volumes,
computed on the DFT grid.

A shearlet system splits an image by scale, like a wavelet, and each scale
further by direction. Each subband then holds the edges of one scale and one
direction, so images with curved edges have sparser coefficients than with
separable wavelets. Every subband has the size of the input. The package has
three systems: [`BandlimitedShearlets`](@ref) (2-D and 3-D),
[`NSSTShearlets`](@ref) (2-D) and [`CompactShearlets`](@ref) (2-D and 3-D);
[Which system](@ref) compares them.

## Installation

Shearlets.jl and its dependencies MultiscaleFilterBanks.jl and
MultiscaleTransformsBase.jl are not registered yet. Clone the three repositories
side by side and develop them into your environment:

```bash
git clone https://github.com/hakkelt/MultiscaleTransformsBase.jl MultiscaleTransformsBase
git clone https://github.com/hakkelt/MultiscaleFilterBanks.jl MultiscaleFilterBanks
git clone https://github.com/hakkelt/Shearlets.jl Shearlets
```

```julia
using Pkg
Pkg.develop([PackageSpec(path = "MultiscaleTransformsBase"),
             PackageSpec(path = "MultiscaleFilterBanks"),
             PackageSpec(path = "Shearlets")])
```

## Example

`analyze` returns the subbands stacked along a new last axis, and
`subband_layout` says what each one is:

```jldoctest index
julia> using Shearlets, Random

julia> Random.seed!(1); x = randn(256, 256);

julia> sys = BandlimitedShearlets(scales = 4);

julia> c = analyze(sys, x); size(c)
(256, 256, 61)

julia> subband_layout(sys, size(x))[10]
ShearletIndex(scale = 2, cone = 1, shear = (2,))

julia> x ≈ synthesize(sys, c)
true
```

## Contents

- How-to guides: [Common tasks](@ref howto) (workspaces, complex data,
  threads, streaming, GPUs), [Inspect responses and choose parameters](@ref howto_parameters),
  and [Examples](@ref examples) that repeat experiments of the shearlet papers.
- [Explanation](@ref theory): how the three systems are built, with figures and
  references.
- [Reference](@ref "API Reference"): every type and function.

## Related packages

[Contourlets.jl](https://github.com/hakkelt/Contourlets.jl) (2-D contourlets)
and [Surfacelets.jl](https://github.com/hakkelt/Surfacelets.jl) (3-D surfacelets)
are directional transforms of the same family. All three are built on
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl),
whose `FilterPair` and `CDF97` Shearlets re-exports, and implement the
interface of
[MultiscaleTransformsBase.jl](https://github.com/hakkelt/MultiscaleTransformsBase.jl)
(`analyze`, `synthesize`, `make_workspace`, `subband_layout`, ...), whose
names Shearlets re-exports too.
