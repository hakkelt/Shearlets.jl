"""
    ShearletIndex{M}

Where a subband sits in a shearlet system on an `N = M + 1`-dimensional grid:

- `scale`: `0` for the lowpass, else `1, …, J` from coarse to fine;
- `cone`: `0` for the lowpass, else the axis the frequencies of the subband are
  largest along (`1`: the horizontal cone |ω₂| ≤ |ω₁| of a matrix, whose rows
  are axis 1; `2`: the vertical cone);
- `shear`: the shear of each of the other `M` axes against `cone`.

A subband of cone `c` and shear `k` at scale `j` collects the frequencies with
`ω_e / ω_c ≈ -k / 2^(j-1)` for the other axes `e`.  Where a shear reaches the
seam between two cones (`|k| = 2^(j-1)` for band-limited shearlets), the
subband is glued across the seam and listed under the lower cone.
"""
struct ShearletIndex{M}
    scale::Int
    cone::Int
    shear::NTuple{M, Int}
end

_lowpass_index(::Val{N}) where {N} = ShearletIndex{N - 1}(0, 0, ntuple(_ -> 0, Val(N - 1)))

"""
    islowpass(index::ShearletIndex) -> Bool

Whether `index` is the lowpass subband (scale 0).
"""
islowpass(i::ShearletIndex) = i.scale == 0

function Base.show(io::IO, i::ShearletIndex)
    if islowpass(i)
        print(io, "ShearletIndex(lowpass)")
    else
        print(io, "ShearletIndex(scale = ", i.scale, ", cone = ", i.cone, ", shear = ", i.shear, ")")
    end
    return nothing
end
