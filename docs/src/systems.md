# Shearlet systems

All three systems compute the same kind of transform: every subband is a
periodic convolution of the input with a filter, so the transform is invariant
under circular shifts and each subband keeps the size of the input. On the DFT
grid each filter is a real, point-symmetric mask, so a transform is one FFT,
then per subband a masked product and an inverse FFT. They differ in how the
masks are built and in what frame they form.

## Indexing

`shearlet_layout(system, size(x))` lists the subbands in the order the
coefficients stack them. A [`ShearletIndex`](@ref) has a `scale` (0 for the
lowpass, then coarse to fine), a `cone` (the axis the subband's frequencies are
largest along) and a `shear` per remaining axis. All systems share the shear
convention: subband `(cone c, shear k)` collects the frequencies with
``\omega_e / \omega_c \approx -k_e / s``, where ``s`` is the number of shears per
side of the cone at that scale.

## Band-limited shearlets (FFST)

[`BandlimitedShearlets`](@ref) follows S. Häuser and G. Steidl, *Fast Finite
Shearlet Transform: a tutorial*, arXiv:1202.1773. The shearlet of scale ``j``,
cone ``c`` and shear ``k`` is

```math
\hat\psi_{j,c,k}(\omega) = \hat\psi_1\!\left(4^{1-j}\lVert\omega\rVert_\infty\right)
\prod_{e \ne c} \hat\psi_2\!\left(2^{j-1}\frac{\omega_e}{\omega_c} + k_e\right),
```

with Meyer's radial window ``\hat\psi_1`` and angular bump ``\hat\psi_2``. The
windows of a scale and the shears of a cone are partitions of unity, so with the
seam shearlets glued across the cones the system is a **Parseval frame**: the
inverse is the adjoint, and ``\sum \lVert c_i\rVert^2 = \lVert x\rVert^2``.

- `smooth = true` replaces the radial windows by the tensor-product construction
  of K. Guo and D. Labate (*The construction of smooth Parseval frames of
  shearlets*, MMNP 8(1), 2013), which makes the seam shearlets smooth.
- `alpha` gives α-shearlets (F. Voigtlaender and A. Pein, arXiv:1702.03559):
  ``2^{(2-\alpha)(j-1)}`` shears per side at scale ``j``.
- On 3-D data the cones are the three pyramids around the axes; a seam shearlet
  is glued across every pyramid whose element has a parallel central direction.

MATLAB's `shearletSystem` is built after the same tutorial but on its own grid,
with the seam halves kept apart and a final normalisation, so it is not a
bitwise reference for this implementation.

## Nonsubsampled shearlet transform (NSST)

[`NSSTShearlets`](@ref) is G. Easley, D. Labate and W.-Q. Lim, *Sparse
directional image representations using the discrete shearlet transform*, ACHA
25(1), 2008: a nonsubsampled Laplacian pyramid (MultiscaleFilterBanks' à trous
pyramid, the one the NSCT uses) followed by shearing filters on every level. The
shearing filters are Meyer windows on the pseudo-polar slope, sampled on an
``L \times L`` grid and applied as ``L \times L`` spatial filters; they sum to
one, so the inverse adds the directional subbands of a level and inverts the
pyramid.

## Compactly supported shearlets (ShearLab 3D)

[`CompactShearlets`](@ref) is G. Kutyniok, W.-Q. Lim and R. Reisenhofer,
*ShearLab 3D: Faithful digital shearlet transforms based on compactly supported
shearlets*, ACM TOMS 42(1), 2016, with the nonseparable generator of W.-Q. Lim,
*Nonseparable shearlet transform*, IEEE TIP 22(5), 2013: a 1-D wavelet across the
cone, a scaling function along it and a 2-D fan filter, digitally sheared on a
refined grid. The filters are compactly supported in space, and the frame is not
tight; the inverse uses the dual filters ``\hat\psi / \sum|\hat\psi|^2``.

The quadrature mirror filter is ShearLab's 9-tap maximally flat lowpass; the fan
filter is the maximally flat fan of order 4 of A. L. da Cunha, J. Zhou and M. N.
Do (IEEE TIP 15(10), 2006, Table II). The construction follows Definition 3.2 of
the ShearLab 3D paper; on 3-D data, Definitions 3.3–3.4.
