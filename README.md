# Shearlets.jl

Translation-invariant discrete shearlet transforms of images and volumes,
computed on the DFT grid.

```julia
using Shearlets

x = randn(256, 256)
sys = BandlimitedShearlets(scales = 4)        # FFST, a Parseval frame
c = shearlet_forward(x, sys)                  # 256 × 256 × 61
shearlet_layout(sys, size(x))[10]             # ShearletIndex(scale = 2, cone = 1, shear = (2,))
x ≈ shearlet_inverse(c, sys)                  # true
```

## Shearlet systems

| System | Construction | Frame | Dimensions |
|---|---|---|---|
| `BandlimitedShearlets` | Häuser & Steidl's Fast Finite Shearlet Transform: Meyer windows on the DFT grid; `smooth = true` for Guo & Labate's smooth seam elements; `alpha` for α-shearlets | Parseval | 2-D, 3-D |
| `NSSTShearlets` | Easley, Labate & Lim's nonsubsampled shearlet transform: à trous pyramid plus shearing filters | biorthogonal | 2-D |
| `CompactShearlets` | ShearLab 3D's compactly supported digital shearlets (nonseparable generator, digital shear) | dual frame | 2-D, 3-D |

Every subband has the size of the input. Real input gives real subbands;
complex input is transformed as its real and imaginary parts.

## Repeated transforms

```julia
ws = make_shearlet_workspace(Float32, size(x), sys)   # masks, FFT plans, buffers
c = similar(x, Float32, size(x)..., shearlet_nsubbands(sys, size(x)))
shearlet_forward!(c, x, sys; workspace = ws)          # allocation-free
shearlet_inverse!(x, c, sys; workspace = ws)
```

The subbands are computed in parallel over Julia's threads unless
`threading = Disabled()`. For data too large to hold every subband,
`shearlet_forward(f, x, sys)` streams them to `f(i, band)` one at a time.

## GPU

Load `Adapt` and a backend with an FFT library (CUDA, AMDGPU), and move a
workspace to the device:

```julia
using CUDA, Adapt
ws = Adapt.adapt(CuArray, make_shearlet_workspace(Float32, size(x), sys))
c = shearlet_forward(CuArray(x), sys; workspace = ws)
```

## Installation

`MultiscaleFilterBanks.jl` is not registered yet and is expected as a sibling
checkout (`../MultiscaleFilterBanks`). On Julia 1.10 run
`julia --project=. dev/bootstrap.jl` once.
