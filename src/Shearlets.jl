"""
    Shearlets

Discrete shearlet transforms of images and volumes, translation invariant and
computed on the DFT grid.

## Quick reference

| Function | Description |
|----------|-------------|
| `analyze` / `synthesize` (and `!` forms) | Transform and its inverse; a streaming form of each |
| `make_workspace` | Masks, plans and buffers for repeated transforms |
| `subband_layout` / `nsubbands` / `subband` | What each subband is, and where it is in the coefficients |
| `shearlet_spectra` | The frequency responses, for inspection |

The verbs and accessors are those of
[MultiscaleTransformsBase](https://github.com/hakkelt/MultiscaleTransformsBase.jl),
re-exported here; every shearlet system is an `AbstractMultiscaleTransform`.

Shearlet systems:

- [`BandlimitedShearlets`](@ref): Häuser & Steidl's FFST, band-limited Meyer
  shearlets, a Parseval frame;
- [`NSSTShearlets`](@ref): Easley, Labate & Lim's nonsubsampled shearlet
  transform, an à trous pyramid with shearing filters (2-D data only);
- [`CompactShearlets`](@ref): ShearLab 3D's compactly supported digital
  shearlets, with dual-frame reconstruction.

Every system is a set of real, point-symmetric masks on the DFT grid, so a
transform is one FFT plus one inverse FFT per subband.  The coefficients of
`x` form one array of size `(size(x)..., nsubbands)`.
"""
module Shearlets

using FFTW: FFTW, plan_rfft, plan_brfft

using MultiscaleTransformsBase: MultiscaleTransformsBase, AbstractMultiscaleTransform, AbstractTransformWorkspace,
    analyze, analyze!, synthesize, synthesize!, make_workspace, transform, similar_coefficients,
    subband_layout, nsubbands, subband, SubbandVector, SubbandLayout, check_workspace, signal_size

using MultiscaleFilterBanks: MultiscaleFilterBanks,
    FilterPair, CDF97, upsample_filter,
    data_eltype, loop_threaded, with_fftw_threads, nsp_responses, DEFAULT_CHUNK_SAMPLES,
    is_device_array,
    MaskBankEngine, MaskBank, MaskItem, SparseMask, IdentityMask, mask_apply!,
    mask_bank_forward!, mask_bank_inverse!, mask_bank_stream_forward!, mask_bank_stream_inverse!, mask_bank_tune

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
    FilterPair, CDF97,

    # Re-exported from MultiscaleTransformsBase
    AbstractMultiscaleTransform, AbstractTransformWorkspace,
    analyze, analyze!, synthesize, synthesize!, make_workspace, transform, similar_coefficients,
    subband_layout, nsubbands, subband, SubbandVector, SubbandLayout,

    # Systems and indexing
    ShearletSystem, BandlimitedShearlets, NSSTShearlets, CompactShearlets, ShearletIndex, islowpass,
    shearlet_spectra,

    # Transforms
    ShearletWorkspace

end # module
