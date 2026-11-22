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
# The pyramid is MultiscaleFilterBanks' à trous pyramid (`nsp_responses`, the
# one the NSCT uses).  On the DFT grid, the subband of level j and window k is
#
#     irfft(B_j · W_{j,k} · X),        B_j = (Π_{i<j} H_i)(1 − G_j H_j),
#
# and the synthesis is X = Σ_j S_j Σ_k rfft(subband) + S_{J+1} rfft(lowpass) with
# S_j = Π_{i<j} G_i: Σ_k W_{j,k} = 1 and Σ_j S_j B_j + S_{J+1} B_{J+1} = 1 make it
# exact.  The frame is not tight; the inverse is not the adjoint.

"""
    NSSTShearlets(; direction_levels = [3, 3, 4, 4], filter_size = nothing, pyramid = CDF97)

The nonsubsampled shearlet transform (NSST) of Easley, Labate & Lim (2008): a
nonsubsampled Laplacian pyramid with `length(direction_levels)` levels, each
split by shearing filters into `2^direction_levels[j]` directional subbands
(scale `j = 1` is the coarsest).  Every subband has the size of the input.  The
system is defined for 2-D data only; other dimensions throw an `ArgumentError`.

- `direction_levels`: `log₂` of the number of directional subbands per scale,
  coarse to fine (any iterable of integers), each at least `2`: a scale with
  `d` levels has `2^(d-2)` shears per side of each cone and two seam subbands
  glued across the cones.
- `filter_size`: the side `L` of the spatial shearing filters, per scale (a vector)
  or for all scales (an integer).  Larger filters resolve directions better near
  the origin; smaller ones are more local.  `0` samples the windows on the
  image's own DFT grid (the frequency-domain variant).  `L` is clamped to each
  side of the image, so an `L` at least as large as both sides is the
  frequency-domain variant too.  The default is `32` for
  the scales coarser than the two finest, which use `16`.
- `pyramid`: the 1-D filter pair of the à trous pyramid; it must be symmetric
  with an odd number of taps.

The frame is not tight: `synthesize` sums the directional subbands of each
level and inverts the pyramid.

`show` prints the keyword call that rebuilds the system, without `pyramid`
when it is `CDF97`.  A pyramid filter of more than 32 taps prints as its size
and element type, which does not parse back; the `text/plain` display adds its
taps.

# Examples
```jldoctest
julia> using Shearlets

julia> sys = NSSTShearlets(direction_levels = [2, 3], filter_size = 0)
NSSTShearlets(direction_levels = [2, 3], filter_size = [0, 0])

julia> nsubbands(sys, (64, 64))   # lowpass, 2^2 and 2^3 directional subbands
13
```

# References
1. G. Easley, D. Labate, and W.-Q. Lim, "Sparse Directional Image Representations
   Using the Discrete Shearlet Transform", *Applied and Computational Harmonic
   Analysis*, 25(1):25–46, 2008.
   [doi:10.1016/j.acha.2007.09.003](https://doi.org/10.1016/j.acha.2007.09.003)
"""
struct NSSTShearlets <: ShearletSystem
    direction_levels::Vector{Int}
    filter_size::Vector{Int}
    pyramid::FilterPair{Float64}
    function NSSTShearlets(direction_levels::AbstractVector{<:Integer}, filter_size::AbstractVector{<:Integer}, pyramid::FilterPair)
        isempty(direction_levels) && throw(ArgumentError("need at least one level"))
        all(>=(2), direction_levels) || throw(ArgumentError("each level needs at least 2^2 directions, got $(direction_levels)"))
        length(filter_size) == length(direction_levels) ||
            throw(ArgumentError("need one filter size per level, got $(length(filter_size)) for $(length(direction_levels)) levels"))
        all(>=(0), filter_size) || throw(ArgumentError("filter sizes must be non-negative"))
        for (name, f) in (("h", pyramid.h), ("g", pyramid.g))
            (isodd(length(f)) && f ≈ reverse(f)) ||
                throw(ArgumentError("pyramid.$(name) must be symmetric with an odd number of taps"))
        end
        # Symmetric to rounding is made exact: the pyramid responses are real
        # only for exactly symmetric filters.
        sym(f) = (h = Vector{Float64}(f); (h .+ reverse(h)) ./ 2)
        return new(Vector{Int}(direction_levels), Vector{Int}(filter_size), FilterPair{Float64}(sym(pyramid.h), sym(pyramid.g)))
    end
end

function NSSTShearlets(; direction_levels = [3, 3, 4, 4], filter_size = nothing, pyramid::FilterPair = CDF97)
    direction_levels = collect(Int, direction_levels)
    J = length(direction_levels)
    sizes = filter_size === nothing ? [j > J - 2 ? 16 : 32 for j in 1:J] :
        filter_size isa Integer ? fill(Int(filter_size), J) : collect(Int, filter_size)
    return NSSTShearlets(direction_levels, sizes, pyramid)
end

function Base.show(io::IO, sys::NSSTShearlets)
    print(io, "NSSTShearlets(direction_levels = ")
    show(io, sys.direction_levels)
    print(io, ", filter_size = ")
    show(io, sys.filter_size)
    if sys.pyramid != CDF97
        print(io, ", pyramid = FilterPair(")
        _show_taps(io, sys.pyramid.h)
        print(io, ", ")
        _show_taps(io, sys.pyramid.g)
        print(io, ")")
    end
    print(io, ")")
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", sys::NSSTShearlets) =
    _show_with_taps(io, sys, sys.pyramid == CDF97 ? () : ("pyramid.h" => sys.pyramid.h, "pyramid.g" => sys.pyramid.g))

Base.:(==)(a::NSSTShearlets, b::NSSTShearlets) =
    a.direction_levels == b.direction_levels && a.filter_size == b.filter_size && a.pyramid == b.pyramid
Base.hash(a::NSSTShearlets, h::UInt) = hash(a.direction_levels, hash(a.filter_size, hash(a.pyramid, hash(NSSTShearlets, h))))

# Shears per side of each cone at a scale with 2^d directions.
_nsst_shears(d::Int) = 2^(d - 2)

# The shearlets of a scale are the band-limited system's 2-D ones: cone 1 with
# k ∈ -s:s, cone 2 with k ∈ 1-s:s-1, the seam shearlets glued across the cones.
function MultiscaleTransformsBase.subband_layout(sys::NSSTShearlets, ::NTuple{2, Int})
    layout = [_lowpass_index(Val(2))]
    for (j, d) in enumerate(sys.direction_levels), (c, k, _) in _bl_scale_elements(Val(2), _nsst_shears(d))
        push!(layout, ShearletIndex{1}(j, c, k))
    end
    return layout
end

# Sizes of any integer type are converted; any other size is an error.
function MultiscaleTransformsBase.subband_layout(sys::NSSTShearlets, dims::Tuple)
    dims isa Tuple{Vararg{Integer}} && !(dims isa Tuple{Vararg{Int}}) && return subband_layout(sys, map(Int, dims))
    throw(ArgumentError("the NSST supports 2-D data only, got size $(dims)"))
end

# The shearing filters of one scale as masks on the `rfft` grid of `dims`
# (vectors over its linear indices), from windows sampled on an L × L grid.
function _nsst_shearing_masks(dims::NTuple{2, Int}, s::Int, L::Int)
    members = [m for (_, _, m) in _bl_scale_elements(Val(2), s)]
    n = length(members)
    rd = _rfft_dims(dims)
    # Frequency-domain variant: the windows sampled on the image grid itself.
    grid = L == 0 ? dims : (min(L, dims[1]), min(L, dims[2]))
    # The squared angular windows of the band-limited system, which sum to one;
    # the slope is measured in units of each axis' Nyquist frequency.
    function evaluate!(v, f)
        cone, r = _cone_slopes((f[1] / grid[1], f[2] / grid[2]))
        _angular_windows!(v, 0, members, cone, r, s, 1.0)
        v .= v .^ 2
        return 1:n
    end
    W = zeros(grid..., n)
    v, acc = zeros(n), zeros(n)
    for I in CartesianIndices(grid)
        # Mean over the Nyquist aliases: the windows still sum to one.
        _alias_sample!(acc, v, evaluate!, I, grid, false)
        W[I, :] .= acc
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

# The directional subbands of a level share its synthesis mask S_j, so they form
# one group: the inverse sums them in space and pays one FFT per level.
function _shearlet_masks(sys::NSSTShearlets, dims::NTuple{2, Int}, ::Type{Tf}) where {Tf}
    J = length(sys.direction_levels)
    rd = _rfft_dims(dims)
    a1, gh1, s1 = nsp_responses(sys.pyramid, J, dims[1]; bins = 1:rd[1])
    a2, gh2, s2 = nsp_responses(sys.pyramid, J, dims[2])
    realpart(z) = (maximum(abs ∘ imag, z) < 1.0e-10 || throw(ArgumentError("the pyramid response is not real")); real.(z))
    sep(u, v) = vec(realpart(u .* transpose(v)))
    grid(v) = reshape(Tf.(v), rd)
    ana = [grid(sep(a1[J + 1], a2[J + 1]))]
    syn = [grid(sep(s1[J + 1], s2[J + 1]))]
    groups = [1:1]
    # Levels with the same shears and filter size share their shearing masks.
    shearing = Dict{Tuple{Int, Int}, Vector{Vector{Float64}}}()
    for j in 1:J
        lvl = J + 1 - j   # pyramid level 1 is the finest
        B = vec(realpart(a1[lvl] .* transpose(a2[lvl]) .* (1 .- gh1[lvl] .* transpose(gh2[lvl]))))
        first = length(ana) + 1
        s, L = _nsst_shears(sys.direction_levels[j]), sys.filter_size[j]
        for W in get!(() -> _nsst_shearing_masks(dims, s, L), shearing, (s, L))
            push!(ana, grid(B .* W))
        end
        push!(groups, first:length(ana))
        push!(syn, grid(sep(s1[lvl], s2[lvl])))
    end
    return _MaskSet(ana, syn, groups, nothing)
end
