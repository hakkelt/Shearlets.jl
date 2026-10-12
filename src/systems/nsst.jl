# The nonsubsampled shearlet transform (NSST) of Easley, Labate & Lim, "Sparse
# directional image representations using the discrete shearlet transform",
# ACHA 25(1), 2008, §3.3: a nonsubsampled Laplacian pyramid followed, at every
# level, by "shearing filters" — small spatial filters that cut the bandpass
# into directional wedges and sum to the identity (Theorem 3.1).
#
# The shearing filters of a level are built as the paper prescribes: a window
# W̃ on the pseudo-polar slope, sampled on an L × L DFT grid and taken to space,
# where it is an L × L filter.  Here the window is the squared angular bump of
# the band-limited system, W̃ = ψ̂₂(s·ω_e/ω_c + k)², so the windows of a level
# sum to one on every sample and the seam windows glue across the cones.  The
# filter is applied by periodic convolution, i.e. as the DFT of its zero-padded
# taps on the image grid; with `filter_size = 0` it is instead sampled on the
# image grid directly (the paper's frequency-domain implementation, §3.1).
#
# The pyramid is MultiscaleFilterBanks' à trous pyramid (`_nsp_responses`, the
# one the NSCT uses).  On the DFT grid, the subband of level j and window k is
#
#     irfft(B_j · W_{j,k} · X),        B_j = (Π_{i<j} H_i)(1 − G_j H_j),
#
# and the synthesis is X = Σ_j S_j Σ_k rfft(subband) + S_{J+1} rfft(lowpass) with
# S_j = Π_{i<j} G_i: Σ_k W_{j,k} = 1 and Σ_j S_j B_j + S_{J+1} B_{J+1} = 1 make it
# exact.  The frame is not tight; the inverse is not the adjoint.

"""
    NSSTShearlets(; directions = [3, 3, 4, 4], filter_size = nothing, pyramid = CDF97)

The nonsubsampled shearlet transform (NSST) of Easley, Labate & Lim (2008): a
nonsubsampled Laplacian pyramid with `length(directions)` levels, each split by
shearing filters into `2^directions[j]` directional subbands (scale `j = 1` is
the coarsest).  Every subband has the size of the input.

- `directions`: `log₂` of the number of directional subbands per scale, coarse to
  fine, each at least `2`: `2^(d-2)` shears per side of each cone, two seam
  subbands glued across the cones.
- `filter_size`: the side `L` of the spatial shearing filters, per scale (a vector)
  or for all scales (an integer).  Larger filters resolve directions better near
  the origin; smaller ones are more local.  `0` samples the windows on the
  image's own DFT grid (the frequency-domain variant).  The default is `32` for
  the scales coarser than the two finest, which use `16`.
- `pyramid`: the 1-D filter pair of the à trous pyramid; it must be symmetric
  with an odd number of taps.

The frame is not tight: `shearlet_inverse` sums the directional subbands of each
level and inverts the pyramid.
"""
struct NSSTShearlets <: ShearletSystem
    directions::Vector{Int}
    filter_size::Vector{Int}
    pyramid::FilterPair{Float64}
    function NSSTShearlets(directions::AbstractVector{<:Integer}, filter_size::AbstractVector{<:Integer}, pyramid::FilterPair)
        isempty(directions) && throw(ArgumentError("need at least one level"))
        all(>=(2), directions) || throw(ArgumentError("each level needs at least 2^2 directions, got $(directions)"))
        length(filter_size) == length(directions) ||
            throw(ArgumentError("need one filter size per level, got $(length(filter_size)) for $(length(directions)) levels"))
        all(>=(0), filter_size) || throw(ArgumentError("filter sizes must be non-negative"))
        for f in (pyramid.h, pyramid.g)
            (isodd(length(f)) && f ≈ reverse(f)) ||
                throw(ArgumentError("the pyramid filters must be symmetric with an odd number of taps"))
        end
        return new(Vector{Int}(directions), Vector{Int}(filter_size), FilterPair{Float64}(pyramid.h, pyramid.g))
    end
end

function NSSTShearlets(; directions::AbstractVector{<:Integer} = [3, 3, 4, 4], filter_size = nothing, pyramid::FilterPair = CDF97)
    J = length(directions)
    sizes = filter_size === nothing ? [j > J - 2 ? 16 : 32 for j in 1:J] :
        filter_size isa Integer ? fill(Int(filter_size), J) : collect(Int, filter_size)
    return NSSTShearlets(directions, sizes, pyramid)
end

Base.:(==)(a::NSSTShearlets, b::NSSTShearlets) =
    a.directions == b.directions && a.filter_size == b.filter_size && a.pyramid.h == b.pyramid.h && a.pyramid.g == b.pyramid.g

# Shears per side of each cone at a scale with 2^d directions.
_nsst_shears(d::Int) = 2^(d - 2)

function shearlet_layout(sys::NSSTShearlets, ::NTuple{2, Int})
    layout = [_lowpass_index(Val(2))]
    for (j, d) in enumerate(sys.directions)
        s = _nsst_shears(d)
        append!(layout, ShearletIndex{1}(j, 1, (k,)) for k in -s:s)
        append!(layout, ShearletIndex{1}(j, 2, (k,)) for k in (1 - s):(s - 1))
    end
    return layout
end

shearlet_layout(::NSSTShearlets, dims::Tuple) =
    throw(ArgumentError("the NSST supports 2-D data only, got size $(dims)"))

# The squared angular windows of one scale (`s` shears per side) at the
# frequency f, in `shearlet_layout` order of the scale; they sum to one.
function _nsst_windows!(v, f::NTuple{2}, s::Int)
    cone = abs(f[2]) <= abs(f[1]) ? 1 : 2
    r = f[1] == f[2] == 0 ? 0.0 : (cone == 1 ? f[2] / f[1] : f[1] / f[2])
    i = 0
    for (c, ks) in ((1, -s:s), (2, (1 - s):(s - 1)))
        for k in ks
            i += 1
            v[i] = (c == cone || abs(k) == s) ? _meyer_psi2(s * r + k)^2 : 0.0
        end
    end
    return v
end

# The shearing filters of one scale as masks on the `rfft` grid of `dims`
# (vectors over its linear indices), from windows sampled on an L × L grid.
function _nsst_shearing_masks(dims::NTuple{2, Int}, s::Int, L::Int)
    n = 4s
    rd = _rfft_dims(dims)
    # Frequency-domain variant: the windows sampled on the image grid itself.
    grid = L == 0 ? dims : (min(L, dims[1]), min(L, dims[2]))
    W = zeros(grid..., n)
    v = zeros(n)
    acc = zeros(n)
    for I in CartesianIndices(grid)
        f = ntuple(d -> _signed_bin(I[d], grid[d]), 2)
        nyq = ntuple(d -> iseven(grid[d]) && abs(f[d]) == grid[d] ÷ 2, 2)
        fill!(acc, 0.0)
        nrep = 0
        # A Nyquist sample stands for ±n/2: average the windows (linearly, so
        # they still sum to one) to keep them point-symmetric.
        for s1 in (nyq[1] ? (-1, 1) : (1,)), s2 in (nyq[2] ? (-1, 1) : (1,))
            fr = (nyq[1] ? s1 * grid[1] ÷ 2 : f[1], nyq[2] ? s2 * grid[2] ÷ 2 : f[2])
            # The slope is measured in units of each axis' Nyquist frequency.
            acc .+= _nsst_windows!(v, (fr[1] / grid[1], fr[2] / grid[2]), s)
            nrep += 1
        end
        W[I, :] .= acc ./ nrep
    end
    grid == dims && return [vec(W[1:rd[1], :, i]) for i in 1:n]
    # Local variant: the L × L filter taps, zero-padded to the image (a tap at
    # offset ±L/2 of an even L is split between both signs to stay symmetric).
    masks = Vector{Vector{Float64}}(undef, n)
    big = zeros(dims)
    for i in 1:n
        w = real.(FFTW.ifft(view(W, :, :, i)))
        fill!(big, 0.0)
        for I in CartesianIndices(grid)
            o = ntuple(d -> _signed_bin(I[d], grid[d]), 2)
            split = ntuple(d -> iseven(grid[d]) && abs(o[d]) == grid[d] ÷ 2 ? (-o[d], o[d]) : (o[d],), 2)
            share = w[I] / (length(split[1]) * length(split[2]))
            for a in split[1], b in split[2]
                big[mod(a, dims[1]) + 1, mod(b, dims[2]) + 1] += share
            end
        end
        masks[i] = vec(real.(FFTW.rfft(big)))
    end
    return masks
end

function _shearlet_masks(sys::NSSTShearlets, dims::NTuple{2, Int}, ::Type{Tf}) where {Tf}
    layout = shearlet_layout(sys, dims)
    J = length(sys.directions)
    rd = _rfft_dims(dims)
    a1, gh1, s1 = _nsp_responses(sys.pyramid, J, dims[1]; bins = 1:rd[1])
    a2, gh2, s2 = _nsp_responses(sys.pyramid, J, dims[2])
    realpart(z) = (maximum(abs ∘ imag, z) < 1.0e-10 || throw(ArgumentError("the pyramid response is not real")); real.(z))
    sep(u, v) = vec(realpart(u .* transpose(v)))
    masks = _DenseMask{Tf, Vector{Tf}}[]
    push!(masks, _DenseMask(Tf.(sep(a1[J + 1], a2[J + 1])), Tf.(sep(s1[J + 1], s2[J + 1]))))
    for j in 1:J
        lvl = J + 1 - j   # pyramid level 1 is the finest
        B = vec(realpart(a1[lvl] .* transpose(a2[lvl]) .* (1 .- gh1[lvl] .* transpose(gh2[lvl]))))
        S = Tf.(sep(s1[lvl], s2[lvl]))   # shared by the directions of the level
        for W in _nsst_shearing_masks(dims, _nsst_shears(sys.directions[j]), sys.filter_size[j])
            push!(masks, _DenseMask(Tf.(B .* W), S))
        end
    end
    return layout, masks
end
