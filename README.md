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
further by direction. Each subband then holds the edges of one scale and one
direction, so images with curved edges have sparser coefficients than with
separable wavelets. This helps in denoising, edge detection and
compressed-sensing reconstruction. The package has three systems:
band-limited shearlets (`BandlimitedShearlets`, 2-D and 3-D), the
nonsubsampled shearlet transform (`NSSTShearlets`, 2-D) and compactly
supported shearlets (`CompactShearlets`, 2-D and 3-D).

```julia
using Shearlets

x = randn(256, 256)
sys = BandlimitedShearlets(scales = 4)
c = analyze(sys, x)                           # 256 × 256 × 61, one subband per slice
subband_layout(sys, size(x))[10]              # ShearletIndex(scale = 2, cone = 1, shear = (2,))
x ≈ synthesize(sys, c)                        # true
```

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

## Documentation

The [manual](https://hakkelt.github.io/Shearlets.jl/dev/) has how-to guides
(workspaces, threads, GPUs, streaming, choosing parameters, worked examples),
an explanation of the three systems with figures and references, and the API
reference.

## Related packages

[Contourlets.jl](https://github.com/hakkelt/Contourlets.jl) (2-D contourlets)
and [Surfacelets.jl](https://github.com/hakkelt/Surfacelets.jl) (3-D surfacelets)
are directional transforms of the same family. All three are built on
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl)
and implement the interface of
[MultiscaleTransformsBase.jl](https://github.com/hakkelt/MultiscaleTransformsBase.jl)
(`analyze`, `synthesize`, `make_workspace`, `subband_layout`, ...), whose
names Shearlets re-exports.

## License

MIT; see [`LICENSE`](LICENSE). Third-party notices are in [`NOTICE`](NOTICE).

## Disclaimer

This package was created by [Claude](https://claude.com/claude-code), Anthropic's
AI model, under the direction of the author.
