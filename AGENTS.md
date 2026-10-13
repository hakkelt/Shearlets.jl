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

Unlike the contourlet / surfacelet split, one package covers both dimensions:
the cone-adapted construction is the same in 2-D and 3-D (two cones, three
pyramids), with no lattice that only exists in the plane.

The surface shared with the core: `FilterPair`, `CDF97`, and the internals `data_eltype`, `n_tasks`, `foreach_chunk`, `chunked_sum!`,
`loop_threaded`, `with_fftw_threads`, `upsample_filter` and `nsp_responses` (the NSST pyramid), imported in
`src/Shearlets.jl`.

**First checkout**: `MultiscaleFilterBanks.jl` is unregistered and expected as a
sibling directory, wired in through `[sources]`. On Julia 1.10, which does not
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
  `shearlet_layout(sys, dims)` and `_shearlet_masks(sys, dims, Tf) -> _MaskSet`;
  nothing else is system-specific.
- `src/engine/masks.jl` — the mask kinds: `_SparseMask` (support indices,
  for band-limited masks), `_DenseMask` (FIR filters) and `_ProductMask`
  (two broadcastable planes, for 3-D FIR filters whose full masks would not fit
  in memory), and `_MaskSet`: an analysis mask per subband, a synthesis mask per
  group of consecutive subbands, and an optional post-factor of the synthesis
  sum. A Parseval system has singleton groups with the analysis masks; a dual
  frame adds the post-factor `1 / Σ Aᵢ²` instead of a second mask per subband;
  the NSST groups each level under its pyramid mask, so the inverse sums a
  level's subbands in space and pays one FFT per level.
- `src/engine/workspace.jl` — buffers, plans, and the threaded forward/inverse
  loops over subbands.
- `src/transforms/shearlet.jl` — the public API.
- `ext/ShearletsAdaptExt.jl` — the GPU path, `Adapt.adapt_structure` only.

## Invariants

- **Coefficients are one stacked array**, `(size(x)..., nsubbands)`, in
  `shearlet_layout` order (lowpass first, scales coarse to fine). Slice with
  `selectdim(c, ndims(c), i)`.
- **Shear convention is shared by all systems**: subband `(cone c, shear k)` sits
  at the frequencies with `ω_e / ω_c ≈ -k_e / s` (`s` shears per side). Plane-wave
  tests assert the index a wave lands in — perfect reconstruction says nothing
  about directional geometry.
- **Real data gives real subbands**: masks are point-symmetric, and a Nyquist
  sample of an even axis stands for both aliases (root mean square for Parseval
  masks, mean for the NSST's sum-to-one windows).
- **Complex data are two real channels** through the same masks, so the
  transform is real-linear bit for bit: `T(a + ib) == T(a) + iT(b)`.
- **Workspace paths allocate nothing**: `shearlet_forward!` / `shearlet_inverse!`
  with a workspace and `threaded = false` are tested with `@allocated == 0`.
  Linear indexing into a matrix view reshapes its parent and allocates — use the
  `bufs_v` vector aliases.
- **No scalar indexing in the mask kernels** (`_mask_apply!`, `_mask_accumulate!`):
  they are broadcasts, so the device path needs no kernels.
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
- **Its own single-stage engine, not MultiscaleFilterBanks' `MaskBankEngine`.**
  The shared engine is two-stage (pyramid mask × bank channel, dense complex
  masks). Band-limited masks are compactly supported in frequency and stored
  sparsely — dense complex masks would take ~25× the memory — and 3-D compact
  shearlets need factored masks; neither fits that shape. The one grouping that
  pays here, the NSST's shared synthesis mask per level, is the `_MaskSet` groups.
- **Dense products use `@..`** (FastBroadcast), as in the shared engine; it does
  not take indexed views, so the sparse gather/scatter stays Base broadcast.
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
