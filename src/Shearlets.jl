"""
    Shearlets

Discrete shearlet transforms of images, translation invariant and computed on
the DFT grid.

## Quick reference

| Function | Description |
|----------|-------------|
| `shearlet_forward` / `shearlet_inverse` | Transform and its inverse; a streaming form of each |
| `make_shearlet_workspace` | Masks, plans and buffers for repeated transforms |
| `shearlet_layout` / `shearlet_nsubbands` | What each subband is |
| `shearlet_spectra` | The frequency responses, for inspection |

Shearlet systems:

- [`BandlimitedShearlets`](@ref): Häuser & Steidl's FFST, band-limited Meyer
  shearlets, a Parseval frame;
- [`NSSTShearlets`](@ref): Easley, Labate & Lim's nonsubsampled shearlet
  transform, an à trous pyramid with shearing filters;
- [`CompactShearlets`](@ref): ShearLab 3D's compactly supported digital
  shearlets, with dual-frame reconstruction.

Every system is a set of real, point-symmetric masks on the DFT grid, so a
transform is one FFT plus one inverse FFT per subband.  The coefficients of
`x` form one array of size `(size(x)..., nsubbands)`.
"""
module Shearlets

using FFTW: FFTW, plan_rfft, plan_brfft
using LinearAlgebra: mul!
using FastBroadcast: @..

using MultiscaleFilterBanks: MultiscaleFilterBanks,
    FilterPair, CDF97, ThreadingPolicy, Auto, Enabled, Disabled, upsample_filter,
    _data_eltype, _n_tasks, _foreach_chunk, _chunked_sum!, _loop_threaded, _with_fftw_threads, _nsp_responses

# ── Types ─────────────────────────────────────────────────────────────────────
include("types/index.jl")
include("types/system.jl")

# ── Mask engine ───────────────────────────────────────────────────────────────
include("engine/masks.jl")
include("engine/workspace.jl")

# ── Shearlet systems ──────────────────────────────────────────────────────────
include("systems/windows.jl")
include("systems/bandlimited.jl")
include("systems/nsst.jl")
include("systems/compact.jl")

# ── Transforms ────────────────────────────────────────────────────────────────
include("transforms/shearlet.jl")

export
    # Re-exported from MultiscaleFilterBanks
    FilterPair, CDF97, ThreadingPolicy, Auto, Enabled, Disabled,

    # Systems and indexing
    ShearletSystem, BandlimitedShearlets, NSSTShearlets, CompactShearlets, ShearletIndex, islowpass,
    shearlet_layout, shearlet_nsubbands, shearlet_spectra,

    # Transforms
    ShearletWorkspace, make_shearlet_workspace,
    shearlet_forward, shearlet_forward!, shearlet_inverse, shearlet_inverse!

end # module
