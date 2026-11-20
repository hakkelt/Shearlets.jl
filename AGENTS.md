# AGENTS.md — Shearlets.jl

This file guides AI agents (and human contributors) through the conventions that
are *specific to this package*. General Julia practice is not repeated here.

## Package family

```
MultiscaleFilterBanks.jl   (shared core: FilterPair, CDF97, à trous responses, threading)
        ├── Contourlets.jl   (2-D CT / NSCT)
        ├── Surfacelets.jl   (surfacelets, for volumes)
        └── Shearlets.jl     (this package — 2-D and 3-D shearlets)
```

Every transform package also implements the interface of
`MultiscaleTransformsBase.jl` (transform objects, `analyze`/`synthesize`,
`make_workspace`, `subband_layout`/`subband`) and re-exports its names. The
shearlet systems subtype `AbstractMultiscaleTransform`; the test suite runs
`MultiscaleTransformsBase.Testing.test_transform` on each. Define interface
methods on Shearlets' own types only (`ShearletSystem`, `ShearletWorkspace`):
a method on the abstract types would be ambiguous with the interface's fallbacks.

Unlike the contourlet / surfacelet split, one package covers both dimensions:
the cone-adapted construction is the same in 2-D and 3-D (two cones, three
pyramids), with no lattice that only exists in the plane.

The surface shared with the core: `FilterPair`, `CDF97`, the mask-bank engine
(`MaskBankEngine`, `MaskBank`, `MaskItem`, `SparseMask`, `IdentityMask`, the
forward/inverse and streaming entry points, `tune_mask_bank`), and the internals
`data_eltype`, `loop_threaded`, `with_fftw_threads`, `nsp_responses` (the NSST
pyramid) and `DEFAULT_CHUNK_SAMPLES`, imported in `src/Shearlets.jl`.

**First checkout**: `MultiscaleFilterBanks.jl` and `MultiscaleTransformsBase.jl` are
unregistered and expected as sibling directories, wired in through `[sources]`. On Julia 1.10, which does not
read `[sources]`, run `julia --project=. dev/bootstrap.jl` once instead.

## Commit Rules

- **Commit Message Formatting**: Conventional Commits, `type(scope): subject`,
  a blank line, then a detailed description (performance numbers or context).
  Always append a co-author line referencing yourself.
- **Committing Changes**: ONLY commit code upon explicit instruction from the user.

## Structure

Every system is a set of real, point-symmetric masks on the `rfft` grid of the
input; a transform is one FFT, then per subband a masked product and an inverse
FFT. The pieces:

- `src/systems/*.jl` — one file per system. A system implements
  `subband_layout(sys, dims)` and `_shearlet_masks(sys, dims, Tf) -> _MaskSet`;
  nothing else is system-specific.
- `src/engine/masks.jl` — the masks: MultiscaleFilterBanks' `SparseMask`
  (band-limited masks), dense arrays on the grid (FIR filters) and Shearlets'
  own `_ProductMask` (two broadcastable planes, for 3-D FIR filters whose full
  masks would not fit in memory; a host loop, broadcasts on devices), and
  `_MaskSet`: an analysis mask per subband, a synthesis mask per group of
  consecutive subbands (`groups = nothing`: one per subband, the analysis
  masks), and an optional post-factor of the synthesis sum. A dual frame adds
  the post-factor `1 / Σ Aᵢ²` instead of a second mask per subband; the NSST
  groups each level under its pyramid mask, so the inverse sums a level's
  subbands in space (the engine's `IdentityMask`) and pays one FFT per level.
- `src/engine/workspace.jl` — the workspace: a `MaskBankEngine` of the masks
  (`_engine`), the buffers that stage arrays the FFT plans cannot take, and
  the device tuner (`make_workspace(sys, x; tune = true)`).
- `src/transforms/shearlet.jl` — the interface methods (`analyze`, `synthesize`,
  the streaming forms, coefficient accessors) and `shearlet_spectra`.
- `ext/ShearletsAdaptExt.jl` — the GPU path: `Adapt.adapt_structure` (the
  masks moved, the engine rebuilt with batched plans: chunks of subbands per
  FFT) and the device routing of `make_workspace(sys, x)`. Adapt is a dependency of
  MultiscaleTransformsBase, so the extension always loads.

## Invariants

- **Coefficients are one stacked array**, `(size(x)..., nsubbands)`, in
  `subband_layout` order (lowpass first, scales coarse to fine). Slice with
  `subband(sys, c, i)`, i.e. `selectdim(c, ndims(c), i)`.
- **Shear convention is shared by all systems**: subband `(cone c, shear k)` sits
  at the frequencies with `ω_e / ω_c ≈ -k_e / s` (`s` shears per side). Plane-wave
  tests assert the index a wave lands in — perfect reconstruction says nothing
  about directional geometry.
- **Real data gives real subbands**: masks are point-symmetric, and a Nyquist
  sample of an even axis stands for both aliases (root mean square for Parseval
  masks, mean for the NSST's sum-to-one windows).
- **Complex data are two real channels** through the same masks, so the
  transform is real-linear bit for bit: `T(a + ib) == T(a) + iT(b)`.
- **Workspace paths allocate nothing**: `analyze!` / `synthesize!`
  with a workspace (or the workspace-only forms) and `threaded = false` are tested with `@allocated == 0`.
- **The engine's FFT plans read and write caller arrays directly** (real data):
  the input, the output and the coefficient slices. `_plannable` decides; other
  arrays (strided views, other element types) are staged through the
  workspace's `xbuf` / `bbuf`, one subband at a time. Sizes whose Float32
  (Float64) slices are not 16-byte aligned get `FFTW.UNALIGNED` plans.
- **No scalar indexing in the device mask products** (`_ProductMask` outside
  the host loop): they are broadcasts, so the device path needs no kernels.
- **Never call `FFTW.set_num_threads`**: plan inside `with_fftw_threads`.
- **Formatting**: `runic --inplace src ext test`, then
  `JULIA_LOAD_PATH="@relint:@stdlib" julia --startup-file=no scripts/relint.jl src/`.

## Design decisions

- **Band-limited shearlets follow the FFST tutorial, not MATLAB.** MATLAB's
  `shearletSystem` (Wavelet Toolbox) is FFST-like but uses its own grid, keeps
  the seam halves unglued and renormalises to Parseval; it is not a bitwise
  reference. Seam shearlets are glued generically: two cone elements are the
  same shearlet when their central directions are parallel.
- **Compact shearlets follow ShearLab 3D's Definition 3.2.** Checked against
  pyshearlab run as a black box (it is GPL — never read its source): the QMF,
  the cascade and the lowpass agree to rounding, and so does every finest-scale
  shearlet when given the same fan filter. At coarser scales the toolbox reuses
  the finest wedge; the definition is far more directional there, so it wins.
  The default QMF is MATLAB's `maxflat` design (taps from MATLAB itself); the
  fan is da Cunha et al.'s maximally flat order-4 fan (Table II), which is not
  the toolbox's `dmaxflat4`.
- **MultiscaleFilterBanks' `MaskBankEngine` runs the transforms**, as a flat
  engine (no pyramid analysis masks), complex data as two real channels per
  subband. Band-limited masks are `SparseMask`s, 3-D compact shearlets keep
  their factored `_ProductMask` with a host loop (Base's broadcast over the
  singleton axes is 40–70 % slower on the host), and the NSST's groups are the
  engine's `groups` with identity bank synthesis masks. On the host it matched
  Shearlets' former engine within noise (forward and inverse, all systems,
  256² and 64³, real and complex).
- **NSST shearing filters** are Meyer windows on the pseudo-polar slope,
  sampled on an `L × L` grid and applied as periodic convolutions (ELL 2008
  §3.3); `filter_size = 0` is the frequency-domain variant (§3.1).

## Testing

- Self-contained `@testitem`s; each imports what it needs except `Shearlets` and
  `Test`. Seed every random input.
- Tags: `:windows`, `:bandlimited`, `:nsst`, `:compact`, `:threed`, `:workspace`,
  `:quality`, `:jet`, `:gpu`.
- `julia --project=test test/runtests.jl :bandlimited,:nsst` filters by tag or
  name. GPU items run on the SLURM `test` partition (`--gres=gpu:1`); on a host
  without CUDA/AMDGPU they are vacuously true.
- All generated output goes under `.temp/` (git-ignored).

## References

Papers (and their extracted text) live in `/project/c_mrrecon/papers/shearlets/`:
Häuser & Steidl (FFST, arXiv:1202.1773), Easley–Labate–Lim (ACHA 2008),
Kutyniok–Lim–Reisenhofer (ShearLab 3D, arXiv:1402.5670), Lim (IEEE TIP 2010,
2013), Guo & Labate (MMNP 2013), Negi & Labate (IEEE TIP 2012), da Cunha–Zhou–Do
(IEEE TIP 2006), Voigtlaender & Pein (α-shearlets, arXiv:1702.03559).
