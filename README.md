# Shearlets.jl

[![CI](https://github.com/hakkelt/Shearlets.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/hakkelt/Shearlets.jl/actions/workflows/CI.yml)
[![Lint](https://github.com/hakkelt/Shearlets.jl/actions/workflows/lint.yml/badge.svg?branch=master)](https://github.com/hakkelt/Shearlets.jl/actions/workflows/lint.yml)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://hakkelt.github.io/Shearlets.jl/stable/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://hakkelt.github.io/Shearlets.jl/dev/)
[![Coverage](https://codecov.io/gh/hakkelt/Shearlets.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/hakkelt/Shearlets.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Runic](https://img.shields.io/badge/code%20style-runic-blue.svg)](https://github.com/fredrikekre/Runic.jl)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Translation-invariant discrete shearlet transforms of images and volumes,
computed on the DFT grid.

A shearlet system splits an image by scale, like a wavelet, and each scale
further by orientation, using sheared copies of one generator. Every subband
then holds the edges of one scale and one direction, so images with curved
edges are represented far more sparsely than by separable wavelets, which helps
denoising, edge detection and compressed-sensing reconstruction.

```julia
using Shearlets

x = randn(256, 256)
sys = BandlimitedShearlets(scales = 4)        # FFST, a Parseval frame
c = shearlet_forward(x, sys)                  # 256 × 256 × 61
shearlet_layout(sys, size(x))[10]             # ShearletIndex(scale = 2, cone = 1, shear = (2,))
x ≈ shearlet_inverse(c, sys)                  # true
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
complex input is transformed as its real and imaginary parts. The
[theory page](https://hakkelt.github.io/Shearlets.jl/stable/theory/) explains the
constructions with figures, and the
[examples](https://hakkelt.github.io/Shearlets.jl/stable/examples/) reproduce the
papers' experiments.

## Repeated transforms and multithreading

```julia
ws = make_shearlet_workspace(Float32, size(x), sys)   # masks, FFT plans, buffers
c = similar(x, Float32, size(x)..., shearlet_nsubbands(sys, size(x)))
shearlet_forward!(c, x, sys; workspace = ws)          # allocation-free
shearlet_inverse!(x, c, sys; workspace = ws)
```

The subbands are computed in parallel over Julia's threads (start Julia with
`-t auto`) unless `threaded = false`. For data too large to hold every subband,
`shearlet_forward(f, x, sys)` streams them to `f(i, band)` one at a time.

## GPU

The transforms run on GPU backends with an FFT library: CUDA.jl (NVIDIA),
AMDGPU.jl (AMD) and Metal.jl (Apple silicon, `Float32` only). The test suite
runs on CUDA. OpenCL.jl is supported but much slower than the CPU, because it
has no FFT: FFTW runs the FFTs on the CPU on `CLArray`s in shared virtual memory
(OpenCL.jl's default on devices that support it). Load `Adapt` and move a
workspace to the device:

```julia
using CUDA, Adapt
ws = Adapt.adapt(CuArray, make_shearlet_workspace(Float32, size(x), sys))
c = shearlet_forward(CuArray(x), sys; workspace = ws)
```

## Related packages

[Contourlets.jl](https://github.com/hakkelt/Contourlets.jl) (2-D contourlets)
and [Surfacelets.jl](https://github.com/hakkelt/Surfacelets.jl) (3-D surfacelets)
are multidirectional transforms of the same family. All three are built on
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl),
whose `FilterPair` and `CDF97` are re-exported here.

## Development

```bash
julia --project=test test/runtests.jl          # all tests
julia --project=docs docs/make.jl              # build the documentation
```

To work against a local checkout of MultiscaleFilterBanks.jl, put it next to
this one (`../MultiscaleFilterBanks`); on Julia 1.10 run
`julia --project=. dev/bootstrap.jl` once.

## Disclaimer

This package was created by [Claude](https://claude.com/claude-code), Anthropic's
AI model, under the direction of the author.

## License

MIT; see [`LICENSE`](LICENSE).
