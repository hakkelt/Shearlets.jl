# Shearlets.jl

Translation-invariant discrete shearlet transforms of images and volumes,
computed on the DFT grid.

```julia
using Shearlets

x = randn(256, 256)
sys = BandlimitedShearlets(scales = 4)        # FFST, a Parseval frame
c = analyze(sys, x)                           # 256 × 256 × 61
subband_layout(sys, size(x))[10]              # ShearletIndex(scale = 2, cone = 1, shear = (2,))
x ≈ synthesize(sys, c)                        # true
```

## Installation

```julia
using Pkg
Pkg.add("Shearlets")
```

## Shearlet systems

| System | Construction | Frame | Dimensions |
|---|---|---|---|
| `BandlimitedShearlets` | Häuser & Steidl's Fast Finite Shearlet Transform: Meyer windows on the DFT grid; `smooth = true` for Guo & Labate's smooth seam elements; `alpha` for α-shearlets | Parseval | 2-D, 3-D |
| `NSSTShearlets` | Easley, Labate & Lim's nonsubsampled shearlet transform: à trous pyramid plus shearing filters | biorthogonal | 2-D |
| `CompactShearlets` | ShearLab 3D's compactly supported digital shearlets (nonseparable generator, digital shear) | dual frame | 2-D, 3-D |

Every subband has the size of the input. Real input gives real subbands;
complex input is transformed as its real and imaginary parts.

The [theory](@ref theory) page explains the constructions and shows their
elements and frequency tilings as the papers do; the [examples](@ref examples)
reproduce the papers' experiments: edge detection, directional selectivity,
shift invariance, denoising and volumes.

## Repeated transforms

```julia
ws = make_workspace(sys, Float32, size(x))     # masks, FFT plans, buffers
c = similar_coefficients(sys, x)
analyze!(c, sys, x; workspace = ws)            # allocation-free
analyze!(c, ws, x)                             # the same, from the workspace alone
synthesize!(x, sys, c; workspace = ws)
```

The subbands are computed in parallel over Julia's threads unless
`threaded = false`. For data too large to hold every subband,
`analyze(f, sys, x)` streams them to `f(i, band)` one at a time.

## GPU

The transforms run on GPU backends with an FFT library: CUDA.jl (NVIDIA),
AMDGPU.jl (AMD) and Metal.jl (Apple silicon, `Float32` only); the test suite
runs on CUDA. OpenCL.jl is supported but much slower than the CPU, because it
has no FFT: FFTW runs the FFTs on the CPU on `CLArray`s in shared virtual memory
(OpenCL.jl's default on devices that support it). Build the workspace from
device data (the Adapt extension, loaded with the interface, moves it there):

```julia
using CUDA
xg = CuArray(x)
ws = make_workspace(sys, xg)
c = analyze(sys, xg; workspace = ws)
```

On the device the subbands are transformed in chunks, one batched FFT per chunk,
which keeps small grids from being bound by kernel launches; the chunk size
follows from the grid size. `make_workspace(sys, xg; tune = true)` measures the
fastest chunk size on the device instead, in a fraction of a second once compiled.
The chunks' stacks and batched FFT plans take up to about 80 MB of device
memory on top of the coefficients; grids too large to chunk, such as 128³, are
transformed one subband at a time.

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
