# Frequency masks on the `rfft` grid, stored on their support only.
#
# Every shearlet system here is a set of real, point-symmetric masks on the DFT
# grid of the input: subband i is `irfft(Mᵢ · rfft(x))`.  The masks are
# compactly supported in frequency and a point is covered by a handful of them,
# so each mask keeps only the linear indices of its nonzero samples and their
# values.  At 512² with four scales the 61 masks hold about as many samples as
# four dense ones would, and a subband touches only its own support.

"""
    _FrequencyMask{Tf}

A subband's real analysis mask `ana` and synthesis mask `syn` on the `rfft`
grid: subband = `irfft(ana · X)`, and the synthesis adds `syn · rfft(subband)`.
For a Parseval system the two are the same array.

- `_SparseMask`: both stored on the linear indices `idx` of the analysis
  support (a mask compactly supported in frequency);
- `_DenseMask`: both stored in full, as vectors over the linear indices of the
  grid (a mask of FIR filters, nonzero almost everywhere);
- `_ProductMask`: the analysis mask a product of factors that broadcast over the
  grid (each array has singleton axes), and the synthesis mask the analysis mask
  over a shared denominator — a dual frame, for 3-D filters whose full masks
  would not fit in memory.
"""
abstract type _FrequencyMask{Tf} end

struct _SparseMask{Tf, I <: AbstractVector{Int32}, V <: AbstractVector{Tf}} <: _FrequencyMask{Tf}
    idx::I
    ana::V
    syn::V
end

_SparseMask(idx::Vector{Int32}, ana::Vector{Tf}, syn::Vector{Tf} = ana) where {Tf} =
    _SparseMask{Tf, Vector{Int32}, Vector{Tf}}(idx, ana, syn)

struct _DenseMask{Tf, V <: AbstractVector{Tf}} <: _FrequencyMask{Tf}
    ana::V
    syn::V
end

_DenseMask(ana::Vector{Tf}, syn::Vector{Tf} = ana) where {Tf} = _DenseMask{Tf, Vector{Tf}}(ana, syn)

# Number of stored samples.
_nstored(m::_SparseMask) = length(m.idx)
_nstored(m::_DenseMask) = length(m.ana)

struct _ProductMask{Tf, A <: AbstractArray{Tf}, B <: AbstractArray{Tf}, C <: AbstractArray{Tf}, D <: AbstractArray{Tf}} <: _FrequencyMask{Tf}
    f1::A
    f2::B
    f3::C
    den::D      # synthesis = f1 f2 f3 / den (shared by all masks of a system)
end

# Z = s · ana · X with X channel c of the spectra (as the N+1-dimensional `X` and
# as the matrix `Xm`); Zv is Z as a vector.
function _mask_apply!(Z, Zv, m::_SparseMask, X, Xm, c::Int, s)
    fill!(Z, zero(eltype(Z)))
    view(Zv, m.idx) .= (m.ana .* s) .* view(Xm, m.idx, c)
    return Z
end

_mask_apply!(Z, Zv, m::_DenseMask, X, Xm, c::Int, s) = (Zv .= (m.ana .* s) .* view(Xm, :, c); Z)

function _mask_apply!(Z, Zv, m::_ProductMask, X, Xm, c::Int, s)
    Z .= (s .* m.f1 .* m.f2 .* m.f3) .* selectdim(X, ndims(X), c)
    return Z
end

# Channel c of the accumulator (the N+1-dimensional `a`, or the matrix `am`) += s · syn · F.
_mask_accumulate!(a, am, m::_SparseMask, F, Fv, c::Int, s) = (view(am, m.idx, c) .+= (m.syn .* s) .* view(Fv, m.idx); am)
_mask_accumulate!(a, am, m::_DenseMask, F, Fv, c::Int, s) = (view(am, :, c) .+= (m.syn .* s) .* Fv; am)

function _mask_accumulate!(a, am, m::_ProductMask, F, Fv, c::Int, s)
    selectdim(a, ndims(a), c) .+= (s .* m.f1 .* m.f2 .* m.f3 ./ m.den) .* F
    return a
end

# Both masks as full arrays on the `rfft` grid `rd`.
function _dense_masks(m::_SparseMask{Tf}, rd::Tuple) where {Tf}
    ana, syn = zeros(Tf, rd), zeros(Tf, rd)
    ana[m.idx] .= m.ana
    syn[m.idx] .= m.syn
    return ana, syn
end

_dense_masks(m::_DenseMask, rd::Tuple) = (reshape(m.ana, rd), reshape(m.syn, rd))

function _dense_masks(m::_ProductMask{Tf}, rd::Tuple) where {Tf}
    ana = zeros(Tf, rd)
    ana .= m.f1 .* m.f2 .* m.f3
    return ana, ana ./ m.den
end

# The `rfft` grid of real data of size `dims`: the first axis halved.
_rfft_dims(dims::NTuple{N, Int}) where {N} = ntuple(d -> d == 1 ? dims[1] ÷ 2 + 1 : dims[d], Val(N))

# Signed frequency, in bins, of 1-based index `i` on an `n`-point DFT axis.
_signed_bin(i::Int, n::Int) = (f = i - 1; f <= n ÷ 2 ? f : f - n)

"""
    _pointwise_masks(evaluate!, nmasks, dims, Tf) -> Vector{_SparseMask{Tf}}

Sample `nmasks` masks given pointwise on the `rfft` grid of a `dims` array.
`evaluate!(v, f)` writes the values of all masks at the frequency `f` (an
`NTuple` of bin offsets, `f[d] ∈ [-dims[d]/2, dims[d]/2]`) into `v`.

The masks must be point-symmetric, `M(-f) = M(f)`, for a real input to give real
subbands.  A sample on the Nyquist plane of an even axis stands for two
frequencies, `+n/2` and `-n/2`, which a mask need not treat alike; there the
mask takes the root mean square of its values over all such aliases.  That keeps
it symmetric on the grid, and keeps `Σᵢ Mᵢ² = 1` wherever the continuous masks
have it, so a Parseval system stays Parseval.
"""
function _pointwise_masks(evaluate!::F, nmasks::Int, dims::NTuple{N, Int}, ::Type{Tf}) where {F, N, Tf}
    rd = _rfft_dims(dims)
    idx = [Int32[] for _ in 1:nmasks]
    val = [Tf[] for _ in 1:nmasks]
    acc = zeros(nmasks)
    v = zeros(nmasks)
    for (lin, I) in enumerate(CartesianIndices(rd))
        f = ntuple(d -> d == 1 ? I[1] - 1 : _signed_bin(I[d], dims[d]), Val(N))
        nyq = ntuple(d -> iseven(dims[d]) && abs(f[d]) == dims[d] ÷ 2, Val(N))
        fill!(acc, 0.0)
        nrep = 0
        # Each Nyquist axis has the aliases ±n/2 (index 1 → -, 2 → +).
        for s in CartesianIndices(ntuple(d -> nyq[d] ? 2 : 1, Val(N)))
            fr = ntuple(d -> nyq[d] ? (2s[d] - 3) * (dims[d] ÷ 2) : f[d], Val(N))
            evaluate!(v, fr)
            acc .+= v .^ 2
            nrep += 1
        end
        for i in 1:nmasks
            a = acc[i]
            if a > 0
                push!(idx[i], Int32(lin))
                push!(val[i], Tf(sqrt(a / nrep)))
            end
        end
    end
    return [_SparseMask(idx[i], val[i]) for i in 1:nmasks]
end
