# Frequency masks on the `rfft` grid.
#
# Every shearlet system here is a set of real, point-symmetric masks on the DFT
# grid of the input: subband i is `irfft(Aᵢ · rfft(x))`, and the synthesis is
#
#     x = irfft(P · Σ_g S_g · rfft(Σ_{i ∈ g} subband_i)),
#
# over groups g of consecutive subbands that share a synthesis mask S_g, with an
# optional post-factor P common to all of them.  A Parseval system has one
# subband per group and S_g = A_g; a dual frame has S_g = A_g and P = 1 / Σ Aᵢ²,
# so neither stores a second mask per subband; the NSST's directional subbands
# of a level share the level's pyramid mask, so a level costs one FFT.

"""
    _FrequencyMask{Tf}

A real mask on the `rfft` grid:

- `_SparseMask`: values `val` on the linear indices `idx` of its support (a mask
  compactly supported in frequency: at 512² with four scales the 61 band-limited
  masks hold about as many samples as four dense ones);
- `_DenseMask`: values over every linear index of the grid (FIR filters);
- `_ProductMask`: the product of two factors that broadcast over the grid (each
  has singleton axes), for 3-D FIR filters whose full masks would not fit in
  memory.
"""
abstract type _FrequencyMask{Tf} end

struct _SparseMask{Tf, I <: AbstractVector{Int32}, V <: AbstractVector{Tf}} <: _FrequencyMask{Tf}
    idx::I
    val::V
end

_SparseMask(idx::Vector{Int32}, val::Vector{Tf}) where {Tf} = _SparseMask{Tf, Vector{Int32}, Vector{Tf}}(idx, val)

struct _DenseMask{Tf, V <: AbstractVector{Tf}} <: _FrequencyMask{Tf}
    val::V
end

_DenseMask(val::Vector{Tf}) where {Tf} = _DenseMask{Tf, Vector{Tf}}(val)

struct _ProductMask{Tf, A <: AbstractArray{Tf}, B <: AbstractArray{Tf}} <: _FrequencyMask{Tf}
    f1::A
    f2::B
end

"""
    _MaskSet(ana, syn, groups, post)

The masks of a system: the analysis mask of every subband, the synthesis mask of
every group of consecutive subbands (`groups`, in order, covering them all), and
the post-factor of the synthesis sum (`nothing` for none).
`_MaskSet(ana; post)` has a group per subband with the analysis masks.
"""
struct _MaskSet{M <: _FrequencyMask, P}
    ana::Vector{M}
    syn::Vector{M}
    groups::Vector{UnitRange{Int}}
    post::P
end

_MaskSet(ana::Vector; post = nothing) = _MaskSet(ana, ana, [i:i for i in eachindex(ana)], post)

# The dense products use FastBroadcast's `@..`: Base's broadcast checks every axis
# for a singleton at run time, which keeps them from vectorising.  Device arrays
# fall back to Base's broadcast.  (`@..` does not take indexed views, so the
# sparse gather/scatter stays Base's.)

# Z = s · A · X with X channel c of the spectra (as the N+1-dimensional `X` and
# as the matrix `Xm`); Zv is Z as a vector.
function _mask_apply!(Z, Zv, m::_SparseMask, X, Xm, c::Int, s)
    fill!(Z, zero(eltype(Z)))
    view(Zv, m.idx) .= (m.val .* s) .* view(Xm, m.idx, c)
    return Z
end

function _mask_apply!(Z, Zv, m::_DenseMask, X, Xm, c::Int, s)
    v, Xc = m.val, view(Xm, :, c)
    @.. Zv = (v * s) * Xc
    return Z
end

function _mask_apply!(Z, Zv, m::_ProductMask, X, Xm, c::Int, s)
    Z .= (s .* m.f1 .* m.f2) .* selectdim(X, ndims(X), c)
    return Z
end

# Channel c of the accumulator (the N+1-dimensional `a`, or the matrix `am`) += s · S · F.
function _mask_accumulate!(a, am, m::_SparseMask, F, Fv, c::Int, s)
    view(am, m.idx, c) .+= (m.val .* s) .* view(Fv, m.idx)
    return am
end

function _mask_accumulate!(a, am, m::_DenseMask, F, Fv, c::Int, s)
    v, ac = m.val, view(am, :, c)
    @.. ac += (v * s) * Fv
    return am
end

function _mask_accumulate!(a, am, m::_ProductMask, F, Fv, c::Int, s)
    selectdim(a, ndims(a), c) .+= (s .* m.f1 .* m.f2) .* F
    return a
end

# On the host the products of `_ProductMask`s are loops: Base's broadcast over
# the singleton axes of the factors does not vectorise, nor does a complex
# product, so the complex arrays are read as pairs of reals (at 64³: 0.17 ms
# against 0.4–0.7 ms for the broadcast).  The first axis is innermost, and a
# factor without it is a scalar along it.
const _HostProduct{Tf} = _ProductMask{Tf, Array{Tf, 3}, Array{Tf, 3}}

function _mask_apply!(Z::Array{Complex{Tf}, 3}, Zv, m::_HostProduct{Tf}, X::Array{Complex{Tf}, 4}, Xm, c::Int, s) where {Tf}
    _product_loop!((z, v) -> v, reinterpret(reshape, Tf, Z), (), m, reinterpret(reshape, Tf, X), (c,), s)
    return Z
end

function _mask_accumulate!(a::Array{Complex{Tf}, 4}, am, m::_HostProduct{Tf}, F::Array{Complex{Tf}, 3}, Fv, c::Int, s) where {Tf}
    _product_loop!(+, reinterpret(reshape, Tf, a), (c,), m, reinterpret(reshape, Tf, F), (), s)
    return a
end

# Zr[:, I, zc...] = op(Zr[:, I, zc...], s · f1[I] · f2[I] · Yr[:, I, yc...]) over
# the 3-D grid I, with Zr and Yr complex arrays read as reals, the factors
# broadcast.
function _product_loop!(op::O, Zr, zc::Tuple, m::_ProductMask, Yr, yc::Tuple, s) where {O}
    f1, f2 = m.f1, m.f2
    size(f2, 1) == 1 && size(f1, 1) != 1 && return _product_loop!(op, Zr, zc, _ProductMask(f2, f1), Yr, yc, s)
    _, n1, n2, n3 = size(Zr)
    sq(f, d, i) = size(f, d) == 1 ? 1 : i
    @inbounds for k in 1:n3, j in 1:n2
        j1, k1, j2, k2 = sq(f1, 2, j), sq(f1, 3, k), sq(f2, 2, j), sq(f2, 3, k)
        if size(f1, 1) != 1
            @simd for i in 1:n1
                _product_pair!(op, Zr, zc, Yr, yc, i, j, k, s * f1[i, j1, k1] * f2[i, j2, k2])
            end
        elseif size(f2, 1) != 1
            a = s * f1[1, j1, k1]
            @simd for i in 1:n1
                _product_pair!(op, Zr, zc, Yr, yc, i, j, k, a * f2[i, j2, k2])
            end
        else
            w = s * f1[1, j1, k1] * f2[1, j2, k2]
            @simd for i in 1:n1
                _product_pair!(op, Zr, zc, Yr, yc, i, j, k, w)
            end
        end
    end
    return Zr
end

@inline function _product_pair!(op::O, Zr, zc, Yr, yc, i, j, k, w) where {O}
    @inbounds Zr[1, i, j, k, zc...] = op(Zr[1, i, j, k, zc...], w * Yr[1, i, j, k, yc...])
    @inbounds Zr[2, i, j, k, zc...] = op(Zr[2, i, j, k, zc...], w * Yr[2, i, j, k, yc...])
    return nothing
end

# The mask as a full array on the `rfft` grid `rd`.
function _dense_mask(m::_SparseMask{Tf}, rd::Tuple) where {Tf}
    out = zeros(Tf, rd)
    out[m.idx] .= m.val
    return out
end

_dense_mask(m::_DenseMask, rd::Tuple) = copy(reshape(m.val, rd))
_dense_mask(m::_ProductMask{Tf}, rd::Tuple) where {Tf} = (out = zeros(Tf, rd); out .= m.f1 .* m.f2; out)

# The `rfft` grid of real data of size `dims`: the first axis halved.
_rfft_dims(dims::NTuple{N, Int}) where {N} = ntuple(d -> d == 1 ? dims[1] ÷ 2 + 1 : dims[d], Val(N))

# Signed frequency, in bins, of 1-based index `i` on an `n`-point DFT axis.
_signed_bin(i::Int, n::Int) = (f = i - 1; f <= n ÷ 2 ? f : f - n)

"""
    _alias_sample!(acc, v, evaluate!, I, grid, rms) -> range

Sample masks given pointwise at the grid point `I` of a DFT grid of size `grid`
(the frequency in signed bins), into `acc`.  `evaluate!(v, f)` writes the masks
at the frequency `f` (an `NTuple` of bin offsets) into `v` and returns the range
of entries it wrote; the others are zero.  Returns the range of `acc` set.

A sample on the Nyquist plane of an even axis stands for two frequencies, `+n/2`
and `-n/2`, which a mask need not treat alike; there each mask takes the root
mean square (`rms`) or the mean of its values over all such aliases.  That keeps
it point-symmetric on the grid, and keeps `Σᵢ Mᵢ² = 1` (root mean square) or
`Σᵢ Mᵢ = 1` (mean) wherever the continuous masks have it.
"""
function _alias_sample!(acc, v, evaluate!::F, I::CartesianIndex{N}, grid::NTuple{N, Int}, rms::Bool) where {F, N}
    f = ntuple(d -> _signed_bin(I[d], grid[d]), Val(N))
    nyq = ntuple(d -> iseven(grid[d]) && abs(f[d]) == grid[d] ÷ 2, Val(N))
    if !any(nyq)
        r = evaluate!(v, f)
        for i in r
            acc[i] = v[i]
        end
        return r
    end
    lo, hi, nrep = typemax(Int), 0, 0
    # Each Nyquist axis has the aliases ±n/2 (index 1 → -, 2 → +).
    for s in CartesianIndices(ntuple(d -> nyq[d] ? 2 : 1, Val(N)))
        fr = ntuple(d -> nyq[d] ? (2s[d] - 3) * (grid[d] ÷ 2) : f[d], Val(N))
        r = evaluate!(v, fr)
        isempty(r) && (nrep += 1; continue)
        # Clear the entries the range grows by.
        nlo, nhi = min(lo, first(r)), max(hi, last(r))
        for i in nlo:nhi
            (i < lo || i > hi) && (acc[i] = 0.0)
        end
        lo, hi = nlo, nhi
        # Entries of the range that this alias did not write are zero there.
        for i in r
            acc[i] += rms ? v[i]^2 : v[i]
        end
        nrep += 1
    end
    hi == 0 && return 1:0
    for i in lo:hi
        acc[i] = rms ? sqrt(acc[i] / nrep) : acc[i] / nrep
    end
    return lo:hi
end

"""
    _pointwise_masks(evaluate!, nmasks, dims, Tf) -> Vector{_SparseMask{Tf}}

Sample `nmasks` point-symmetric masks given pointwise (see `_alias_sample!`,
with root mean squares on the Nyquist planes) on the `rfft` grid of a `dims`
array, storing each on its support.
"""
function _pointwise_masks(evaluate!::F, nmasks::Int, dims::NTuple{N, Int}, ::Type{Tf}) where {F, N, Tf}
    idx = [Int32[] for _ in 1:nmasks]
    val = [Tf[] for _ in 1:nmasks]
    acc = zeros(nmasks)
    v = zeros(nmasks)
    for (lin, I) in enumerate(CartesianIndices(_rfft_dims(dims)))
        for i in _alias_sample!(acc, v, evaluate!, I, dims, true)
            if acc[i] > 0
                push!(idx[i], Int32(lin))
                push!(val[i], Tf(acc[i]))
            end
        end
    end
    return [_SparseMask(idx[i], val[i]) for i in 1:nmasks]
end
