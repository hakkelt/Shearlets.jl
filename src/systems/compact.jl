# Compactly supported digital shearlets — Kutyniok, Lim & Reisenhofer,
# "ShearLab 3D: Faithful digital shearlet transforms based on compactly
# supported shearlets", ACM TOMS 42(1), 2016 (arXiv:1402.5670), §3.2 and §3.4;
# the nonseparable generator is W.-Q. Lim, "Nonseparable shearlet transform",
# IEEE TIP 22(5), 2013.
#
# The generator is ψ̂(ξ) = P(ξ₁/2, ξ₂) ψ̂₁(ξ₁) φ̂₁(ξ₂): a 1-D wavelet across the
# cone, a 1-D scaling function along it, and a 2-D fan filter P that cuts the
# result to a wedge.  Digitally (Definition 3.2, with the sampling constants of
# §3.4 so that nothing is subsampled), the filter of scale ℓ ∈ 1:J (coarse to
# fine) with shear level d = d_ℓ, before shearing, is
#
#     ψ_ℓ = p_ℓ ∗ (g_{J+1−ℓ} ⊗ h_{J+1−ℓ+d}),
#
# with h_j, g_j the j-level à trous lowpass and wavelet filters of a quadrature
# mirror pair (eq. 5) and p_ℓ the taps of P(2^{J−ℓ} ξ₁, 2^{J+1−ℓ+d} ξ₂) (the fan
# filter upsampled).  The shear S_{k/2^d}, |k| ≤ 2^d, does not preserve the
# integer grid, so it is applied faithfully on a grid refined 2^d times across
# the cone (eqs. 21–22): upsample along axis 1 and interpolate with h_d, remap
# the indices by the integer shear S_k, filter with h_d again and downsample.
# The vertical cone uses the transposed filters.
#
# Checked against pyshearlab (run as a black box): the quadrature mirror filter,
# the à trous cascade and the lowpass agree to rounding, and with the same fan
# filter so does every shearlet of the finest scale, sheared or not.  At coarser
# scales the toolbox keeps the finest scale's wedge (the fan upsampled 2^(d+1)
# times along axis 2, with h_{d+1}) instead of Definition 3.2's
# P(2^{J−ℓ} ξ₁, 2^{J+1−ℓ+d} ξ₂) and h_{J+1−ℓ+d}; the definition is markedly more
# directional there (99% against 65% of a scale-1 shearlet's energy inside its
# own angular sector at J = 4), so this follows the paper.  The toolbox's default
# `dmaxflat4` fan is not the maximally flat fan of da Cunha et al.'s Table II.
#
# All filters are finite, so the transform is a periodic convolution with each:
# a mask on the DFT grid (the filter's taps folded onto the image grid and
# transformed).  The frame is not tight; the inverse uses the dual filters
# γ̂ = ψ̂ / Ψ̂ with Ψ̂ = Σ |ψ̂|² over all filters, lowpass included (eqs. 31–33).

# The 9-tap maximally flat lowpass with its 3 dB point at half the band that
# ShearLab 3D uses for its quadrature mirror pair (§5.1): the taps MATLAB R2022a's
# `design(fdesign.lowpass('N,F3dB', 8, 0.5), 'maxflat')` returns, each mirrored
# pair averaged to make the filter exactly symmetric.
const _SHEARLAB_QMF = [
    0.010493326175840769, -0.026348304703363043, -0.05177669529663706, 0.27634830470336308, 0.58256673824159266,
    0.27634830470336308, -0.05177669529663706, -0.026348304703363043, 0.010493326175840769,
]

# The maximally flat fan filter of order 4 of da Cunha, Zhou & Do (IEEE TIP 15(10),
# 2006, Table II): with x = cos ω, F₄(x₁, x₂) = (x₁ − x₂)(35 − 5x₁² + 25x₁x₂
# − 3x₁³x₂ − 5x₂² + 15x₁²x₂² − 3x₁x₂³ + 5x₁³x₂³)/32 is −1 on the horizontal fan
# |ω₂| < |ω₁| (F₄(π, 0) = −1) and +1 on the vertical one, so P = (1 − F₄)/2
# passes the horizontal fan and P + P' = 1
# with P' the transposed filter.  Returns the 9 × 9 taps, origin at the centre.
function _maxflat_fan()
    F(x1, x2) = (x1 - x2) * (35 - 5x1^2 + 25x1 * x2 - 3x1^3 * x2 - 5x2^2 + 15x1^2 * x2^2 - 3x1 * x2^3 + 5x1^3 * x2^3) / 32
    n = 16   # P is a trigonometric polynomial of degree 4 per axis: 16 samples are exact
    P = [(1 - F(cospi(2a / n), cospi(2b / n))) / 2 for a in 0:(n - 1), b in 0:(n - 1)]
    taps = real.(FFTW.ifft(P))
    idx = mod.(-4:4, n) .+ 1
    return taps[idx, idx]
end

"""
    CompactShearlets(; scales = 4, shear_levels = ceil.(Int, (1:scales) ./ 2), qmf, fan)

The compactly supported digital shearlets of ShearLab 3D (Kutyniok, Lim &
Reisenhofer, 2016), without subsampling: shift invariant, every subband the size
of the input.  Spatially compact filters, at the price of a frame that is not
tight — the inverse uses the dual filters.

- `scales`: number of scales `J`, coarse to fine.
- `shear_levels`: per scale the level `d`: shears `k = -2^d, …, 2^d` in each
  cone, `2(2^(d+1) + 1)` subbands per scale.  The default is parabolic scaling.
- `qmf`: the 1-D lowpass of the quadrature mirror pair (symmetric, odd length);
  the highpass is its modulation `(-1)^n h[n]`.  The default is ShearLab's
  9-tap maximally flat filter.
- `fan`: the 2-D fan filter `P` as a matrix of taps with the origin at its
  centre, passing the horizontal fan `|ω₂| < |ω₁|`; the default is the
  maximally flat fan filter of order 4 of da Cunha, Zhou & Do (2006).

The two cones overlap at the seam lines (`|k| = 2^d` in both), as in ShearLab.

On 3-D data the cones are the three pyramids around the axes and a shearlet has
a shear `(k₁, k₂)` against each of the other two axes, `3(2^(d+1) + 1)²`
subbands per scale (ShearLab 3D's Definitions 3.3–3.4).
"""
struct CompactShearlets <: ShearletSystem
    scales::Int
    shear_levels::Vector{Int}
    qmf::Vector{Float64}
    fan::Matrix{Float64}
    function CompactShearlets(scales::Integer, shear_levels::AbstractVector{<:Integer}, qmf::AbstractVector{<:Real}, fan::AbstractMatrix{<:Real})
        scales >= 1 || throw(ArgumentError("need at least one scale, got $(scales)"))
        length(shear_levels) == scales ||
            throw(ArgumentError("need one shear level per scale, got $(length(shear_levels)) for $(scales) scales"))
        all(>=(0), shear_levels) || throw(ArgumentError("shear levels must be non-negative"))
        (isodd(length(qmf)) && qmf ≈ reverse(qmf)) ||
            throw(ArgumentError("the quadrature mirror filter must be symmetric with an odd number of taps"))
        (all(isodd, size(fan)) && fan ≈ reverse(fan)) ||
            throw(ArgumentError("the fan filter must be point-symmetric with odd sides"))
        return new(Int(scales), Vector{Int}(shear_levels), Vector{Float64}(qmf), Matrix{Float64}(fan))
    end
end

CompactShearlets(;
    scales::Integer = 4, shear_levels::AbstractVector{<:Integer} = ceil.(Int, (1:scales) ./ 2),
    qmf::AbstractVector{<:Real} = _SHEARLAB_QMF, fan::AbstractMatrix{<:Real} = _maxflat_fan(),
) = CompactShearlets(scales, shear_levels, qmf, fan)

Base.:(==)(a::CompactShearlets, b::CompactShearlets) =
    a.scales == b.scales && a.shear_levels == b.shear_levels && a.qmf == b.qmf && a.fan == b.fan
Base.hash(a::CompactShearlets, h::UInt) =
    hash(a.scales, hash(a.shear_levels, hash(a.qmf, hash(a.fan, hash(CompactShearlets, h)))))

function shearlet_layout(sys::CompactShearlets, dims::NTuple{N, Int}) where {N}
    N in (2, 3) || throw(ArgumentError("compactly supported shearlets support 2-D and 3-D data, got size $(dims)"))
    layout = [_lowpass_index(Val(N))]
    for (ℓ, d) in enumerate(sys.shear_levels), cone in 1:N, kI in CartesianIndices(ntuple(_ -> -(2^d):(2^d), Val(N - 1)))
        push!(layout, ShearletIndex{N - 1}(ℓ, cone, Tuple(kI)))
    end
    return layout
end

shearlet_layout(::CompactShearlets, dims::Tuple) =
    throw(ArgumentError("compactly supported shearlets support 2-D and 3-D data, got size $(dims)"))

# ── Centred filter algebra ────────────────────────────────────────────────────
# Filters are arrays with odd sides and the origin at the centre.

_centre(n::Int) = (n + 1) ÷ 2

# Insert r − 1 zeros between the taps along `dim`.
function _upsample(a::AbstractArray, r::Int, dim::Int)
    r == 1 && return copy(a)
    sz = ntuple(d -> d == dim ? (size(a, d) - 1) * r + 1 : size(a, d), ndims(a))
    out = zeros(eltype(a), sz)
    selectdim(out, dim, 1:r:sz[dim]) .= a
    return out
end

# Keep the taps at offsets that are multiples of r along `dim`.
function _downsample(a::AbstractArray, r::Int, dim::Int)
    r == 1 && return copy(a)
    c = _centre(size(a, dim))
    m = (c - 1) ÷ r
    return copy(selectdim(a, dim, (c - m * r):r:(c + m * r)))
end

# Full linear convolution of two centred arrays of the same dimension, by real
# FFTs of a size with small prime factors (these sizes are otherwise arbitrary,
# and an FFT of a large prime size is several times slower).
function _conv(a::AbstractArray, b::AbstractArray)
    sz = size(a) .+ size(b) .- 1
    n = map(m -> nextprod((2, 3, 5), m), sz)
    pa = zeros(n); pa[CartesianIndices(size(a))] .= a
    pb = zeros(n); pb[CartesianIndices(size(b))] .= b
    full = FFTW.irfft(FFTW.rfft(pa) .* FFTW.rfft(pb), n[1])
    return full[CartesianIndices(sz)]
end

# A 1-D filter oriented along `dim` of an N-dimensional array.
_along(f::AbstractVector, dim::Int, N::Int) = reshape(f, ntuple(d -> d == dim ? length(f) : 1, N))

# The j-level à trous lowpass: ĥ_j(ξ) = Π_{k<j} ĥ(2^k ξ) (eq. 5), ĥ_0 = 1.
function _atrous_lowpass(h::Vector{Float64}, j::Int)
    out = [1.0]
    for k in 0:(j - 1)
        out = _conv(out, upsample_filter(h, 2^k))
    end
    return out
end

# The j-level wavelet filter: ĝ_j(ξ) = ĝ(2^(j−1) ξ) ĥ_{j−1}(ξ), with the highpass
# of the mirror pair g[n] = (−1)^n h[n] (n the offset from the centre).
function _atrous_highpass(h::Vector{Float64}, j::Int)
    c = _centre(length(h))
    g = [(-1)^(n - c) * h[n] for n in eachindex(h)]
    return _conv(upsample_filter(g, 2^(j - 1)), _atrous_lowpass(h, j - 1))
end

# The digital shears S_{k/2^d}, k = -2^d, …, 2^d, of a 2-D filter across axis 1
# (eqs. 21–22): refine axis 1 by r = 2^d with the interpolator h_d, apply the
# integer shear (n₁, n₂) ↦ (n₁ − k n₂, n₂) on the refined grid, filter with h_d
# again and return to the original grid.  Shear k sends the passband to the
# slope ω₂/ω₁ = −k/2^d, as in the other systems.
function _digital_shears(f::Matrix{Float64}, d::Int, h::Vector{Float64})
    r = 2^d
    hd = _atrous_lowpass(h, d)
    fine = _conv(_upsample(f, r, 1), _along(r .* hd, 1, 2))
    n1, n2 = size(fine)
    c1, c2 = _centre(n1), _centre(n2)
    return map(-r:r) do k
        pad = abs(k) * (c2 - 1)
        out = zeros(n1 + 2pad, n2)
        co = c1 + pad
        for j in 1:n2, i in 1:n1
            # out[o₁, o₂] = fine[o₁ − k o₂, o₂]: the sheared filter's offsets.
            out[co + (i - c1) + k * (j - c2), j] = fine[i, j]
        end
        _downsample(_conv(out, _along(hd, 1, 2)), r, 1)
    end
end

# The real frequency response of a centred, point-symmetric filter on a DFT grid
# of size `n` (one entry per axis of `f`): the filter folded onto the periodic
# grid (offset o ↦ o mod n) and transformed, with its axis `h` halved as by
# `rfft` (`h = 0`: none).
function _centred_response(f::AbstractArray, n::Tuple, h::Int)
    folded = zeros(n)
    c = _centre.(size(f))
    for I in CartesianIndices(f)
        folded[CartesianIndex(ntuple(a -> mod(I[a] - c[a], n[a]) + 1, Val(ndims(f))))] += f[I]
    end
    F = h == 0 ? FFTW.fft(folded) : FFTW.rfft(folded, (h, (a for a in 1:length(n) if a != h)...))
    maximum(abs ∘ imag, F) <= 1.0e-9 * max(1.0, maximum(abs, F)) ||
        throw(ArgumentError("shearlet filter is not point-symmetric"))
    return real.(F)
end

# The 2-D wedge of scale ℓ before shearing: p_ℓ ∗ (δ ⊗ h_{J+1−ℓ+d}).
function _compact_wedge(sys::CompactShearlets, ℓ::Int)
    J, d, h = sys.scales, sys.shear_levels[ℓ], sys.qmf
    p = _upsample(_upsample(sys.fan, 2^(J - ℓ), 1), 2^(J + 1 - ℓ + d), 2)
    return _conv(p, transpose(_atrous_lowpass(h, J + 1 - ℓ + d)))
end

# The dual frame: the synthesis divides by Ψ̂ = Σ |ψ̂|² over all filters, applied
# once to the synthesis sum.
function _compact_post(den::AbstractArray, dims::Tuple, ::Type{Tf}) where {Tf}
    minimum(den) > 0 || throw(ArgumentError("the shearlet filters do not cover every frequency of a $(dims) grid"))
    return Tf.(1 ./ den)
end

# Two dimensions: the shearlets of scale ℓ are the digital shears of the
# generator ψ_ℓ = p_ℓ ∗ (g_{J+1−ℓ} ⊗ h_{J+1−ℓ+d}), the wedge times the wavelet
# across the cone; cone 2 is the transpose of cone 1.
function _shearlet_masks(sys::CompactShearlets, dims::NTuple{2, Int}, ::Type{Tf}) where {Tf}
    J, h = sys.scales, sys.qmf
    hJ = _atrous_lowpass(h, J)
    responses = [_centred_response(hJ .* transpose(hJ), dims, 1)]
    for ℓ in 1:J
        ψ = _conv(_compact_wedge(sys, ℓ), _along(_atrous_highpass(h, J + 1 - ℓ), 1, 2))
        sheared = _digital_shears(ψ, sys.shear_levels[ℓ], h)
        append!(responses, _centred_response(f, dims, 1) for f in sheared)
        append!(responses, _centred_response(permutedims(f), dims, 1) for f in sheared)
    end
    post = _compact_post(sum(a -> a .^ 2, responses), dims, Tf)
    return _MaskSet([_DenseMask(Tf.(vec(a))) for a in responses]; post)
end

# ── Three dimensions ─────────────────────────────────────────────────────────
# Definitions 3.3–3.4: the shearlet of pyramid c (dominant axis c, other axes
# e₁ < e₂) with shears (k₁, k₂) is
#
#     ψ̂(ξ) = ĝ_{J+1−ℓ}(ξ_c) · Φ̂_{k₁}(ξ_c, ξ_{e₁}) · Φ̂_{k₂}(ξ_c, ξ_{e₂}),
#
# where Φ_k = S_{k/2^d}(p_ℓ ∗_{x₂} h_{J+1−ℓ+d}) is the 2-D wedge of the planar
# construction without its wavelet, digitally sheared.  A full mask per subband
# would take subbands × voxels samples, so each is kept as two planes
# (`_ProductMask`): ĝ · Φ̂_{k₁} over (c, e₁) and Φ̂_{k₂} over (c, e₂).

# A response over the axes `ax` (one or two, in the response's own order) placed
# on the 3-D `rfft` grid of `dims`: axis 1 halved, the axes not in `ax` singletons.
function _oriented_response(f::AbstractArray, ax::Tuple, dims::NTuple{3, Int})
    R = _centred_response(f, map(a -> dims[a], ax), something(findfirst(==(1), ax), 0))
    if length(ax) == 2 && ax[1] > ax[2]
        R = permutedims(R)
        ax = (ax[2], ax[1])
    end
    return reshape(R, ntuple(a -> (i = findfirst(==(a), ax); i === nothing ? 1 : size(R, i)), 3))
end

function _shearlet_masks(sys::CompactShearlets, dims::NTuple{3, Int}, ::Type{Tf}) where {Tf}
    J, h = sys.scales, sys.qmf
    hJ = _atrous_lowpass(h, J)
    factors = Tuple{Array{Float64, 3}, Array{Float64, 3}}[]
    push!(factors, (_oriented_response(hJ .* transpose(hJ), (1, 2), dims), _oriented_response(hJ, (3,), dims)))
    for ℓ in 1:J
        g = _atrous_highpass(h, J + 1 - ℓ)
        wedges = _digital_shears(_compact_wedge(sys, ℓ), sys.shear_levels[ℓ], h)
        for c in 1:3
            e1, e2 = Tuple(e for e in 1:3 if e != c)
            G = _oriented_response(g, (c,), dims)
            GW1 = [G .* _oriented_response(w, (c, e1), dims) for w in wedges]
            W2 = [_oriented_response(w, (c, e2), dims) for w in wedges]
            for i2 in eachindex(wedges), i1 in eachindex(wedges)
                push!(factors, (GW1[i1], W2[i2]))
            end
        end
    end
    den = zeros(_rfft_dims(dims))
    for (a, b) in factors
        den .+= (a .* b) .^ 2
    end
    # Factors shared between subbands stay shared after the conversion.
    cache = IdDict{Array{Float64, 3}, Array{Tf, 3}}()
    conv(a) = get!(() -> Tf.(a), cache, a)
    return _MaskSet([_ProductMask(conv(a), conv(b)) for (a, b) in factors]; post = _compact_post(den, dims, Tf))
end
