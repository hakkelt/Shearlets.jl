# Shearlets.jl

Translation-invariant discrete shearlet transforms of images and volumes,
computed on the DFT grid.

A shearlet system splits an image by scale, like a wavelet, and each scale
further by orientation, using sheared copies of one generator. Every subband
then holds the edges of one scale and one direction, so images with curved
edges are represented more sparsely than by separable wavelets. This helps
denoising, edge detection and compressed-sensing reconstruction.

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

## Quick example

```jldoctest index
julia> using Shearlets, Random

julia> Random.seed!(1); x = randn(256, 256);

julia> sys = BandlimitedShearlets(scales = 4);   # FFST, a Parseval frame

julia> c = analyze(sys, x); size(c)
(256, 256, 61)

julia> subband_layout(sys, size(x))[10]
ShearletIndex(scale = 2, cone = 1, shear = (2,))

julia> x ≈ synthesize(sys, c)
true
```

## Shearlet systems

| System | Construction | Frame | Dimensions |
|---|---|---|---|
| `BandlimitedShearlets` | Häuser & Steidl's Fast Finite Shearlet Transform: Meyer windows on the DFT grid; `smooth = true` for Guo & Labate's smooth seam elements; `alpha` for α-shearlets | Parseval | 2-D, 3-D |
| `NSSTShearlets` | Easley, Labate & Lim's nonsubsampled shearlet transform: à trous pyramid plus shearing filters | biorthogonal | 2-D |
| `CompactShearlets` | ShearLab 3D's compactly supported digital shearlets (nonseparable generator, digital shear) | dual frame | 2-D, 3-D |

Every subband has the size of the input. Real input gives real subbands;
complex input is transformed as its real and imaginary parts.

## Where to go next

- [How-to guides](@ref howto): repeated transforms, threading, complex data, GPUs,
  and the [examples](@ref examples) that reproduce the papers' experiments.
- [Explanation](@ref theory): the constructions, with figures of their elements
  and frequency tilings, and the references.
- [Reference](@ref "API Reference"): every type and function.

## Related packages

[Contourlets.jl](https://github.com/hakkelt/Contourlets.jl) (2-D contourlets)
and [Surfacelets.jl](https://github.com/hakkelt/Surfacelets.jl) (3-D surfacelets)
are multidirectional transforms of the same family. All three are built on
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl),
whose `FilterPair` and `CDF97` are re-exported here, and implement the common
interface of
[MultiscaleTransformsBase.jl](https://github.com/hakkelt/MultiscaleTransformsBase.jl)
(`analyze`, `synthesize`, `make_workspace`, `subband_layout`, ...), whose names
are re-exported too.
