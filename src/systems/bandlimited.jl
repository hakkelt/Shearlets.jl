# Band-limited cone-adapted shearlets on the full DFT grid — Häuser & Steidl,
# "Fast Finite Shearlet Transform: a tutorial", arXiv:1202.1773, §3.1–3.4, in
# any dimension.
#
# On a frequency ω (scaled so that the Nyquist frequency of every axis is
# X = 2^(2J−1)), the shearlet of scale j ∈ 1:J, cone c and shear k is
#
#     ψ̂ⱼ,c,ₖ(ω) = R_j(ω) · Πₑ ψ̂₂(sⱼ ω_e / ω_c + k_e),    e ≠ c,
#
# on the cone (pyramid, in 3-D) |ω_e| ≤ |ω_c|, where ψ̂₂ is the angular bump
# (`_meyer_psi2`), sⱼ = 2^(j−1) the number of shears per side (2^((2−α)(j−1))
# for α-shearlets, after Voigtlaender & Pein, arXiv:1702.03559) and R_j the
# radial window of scale j:
#
# - Meyer (eqs. 12–16): R_j(ω) = ψ̂₁(4^(1−j) ‖ω‖_∞), lowpass ϕ(‖ω‖_∞);
# - smooth (eqs. 27–28, after Guo & Labate 2013): with the tensor lowpass
#   Φ(ω) = Πd ϕ(ω_d), R_j(ω) = √(Φ(4^(−j) ω)² − Φ(4^(1−j) ω)²), lowpass Φ.
#
# Σⱼ R_j² + lowpass² = 1 telescopes, and the shears of a scale partition each
# cone (Σₖ ψ̂₂(x + k)² = 1 per axis), so with the seam shearlets (some
# |k_e| = sⱼ) glued across the cones the system is a Parseval frame of the finite
# grid: the inverse is the adjoint.  A shearlet of cone c peaks on the direction
# v with v_c = sⱼ, v_e = −k_e; seam shearlets of different cones are the same
# when their directions are parallel, which in 2-D glues equal k (the FFST's
# ψ^{h×v}) and in 3-D also glues the edges and corners of the pyramids.  The
# Meyer seam elements are continuous but not smooth across the seams; the
# smooth variant's are.

"""
    BandlimitedShearlets(; scales = 0, smooth = false, alpha = 1)

The band-limited cone-adapted shearlet system of Häuser & Steidl's Fast Finite
Shearlet Transform (FFST): Meyer-type windows on the full DFT grid, a Parseval
frame whose inverse is its adjoint.  Works on 2-D and 3-D data (cones become
the three pyramids around the axes).

- `scales`: number of scales `J` (`0` picks `⌊log₂(min(size)) / 2⌋`).  Scale `j`
  has `s = 2^(j-1)` shears per side of each cone; in 2-D that is `4s` shearlets,
  two of them seam shearlets glued across the cones.
- `smooth`: use the tensor-product lowpass of Guo & Labate's construction, which
  makes the seam shearlets smooth (and better localised in space) at the price
  of a less cone-aligned lowpass.
- `alpha`: the anisotropy of α-shearlets, `0 ≤ α ≤ 2`: `s = round(2^((2-α)(j-1)))`
  shears per side.  `α = 1` is parabolic scaling (shearlets), `α = 2` keeps
  one shear per side at every scale (wavelet-like), `α → 0` approaches ridgelets.

Every subband has the size of the input; real input gives real subbands.
"""
struct BandlimitedShearlets <: ShearletSystem
    scales::Int
    smooth::Bool
    alpha::Float64
    function BandlimitedShearlets(scales::Integer, smooth::Bool, alpha::Real)
        scales >= 0 || throw(ArgumentError("scales must be non-negative, got $(scales)"))
        0 <= alpha <= 2 || throw(ArgumentError("alpha must lie in [0, 2], got $(alpha)"))
        return new(Int(scales), smooth, Float64(alpha))
    end
end

BandlimitedShearlets(; scales::Integer = 0, smooth::Bool = false, alpha::Real = 1) = BandlimitedShearlets(scales, smooth, alpha)

function _nscales(sys::BandlimitedShearlets, dims::Tuple)
    sys.scales > 0 && return sys.scales
    return max(1, floor(Int, log2(minimum(dims)) / 2))
end

# Shears per side of a cone at scale j.
_bl_shears(sys::BandlimitedShearlets, j::Int) = max(1, round(Int, 2.0^((2 - sys.alpha) * (j - 1))))

# The shearlets of one scale with `s` shears per side on an N-dimensional grid:
# for each, the shear it has in every cone it reaches (`nothing` where it does
# not), listed under the lowest such cone.  Two cone elements are glued when
# their central directions (v_c = s, v_e = −k_e) are parallel; all have
# ‖v‖_∞ = s, so parallel means equal up to sign.
function _bl_scale_elements(::Val{N}, s::Int) where {N}
    M = N - 1
    elements = Tuple{Int, NTuple{M, Int}, Vector{Union{Nothing, NTuple{M, Int}}}}[]
    seen = Dict{NTuple{N, Int}, Int}()
    for c in 1:N, kI in CartesianIndices(ntuple(_ -> -s:s, Val(M)))
        k = Tuple(kI)
        # k shears the other axes in increasing order.
        v = ntuple(d -> d == c ? s : -k[d < c ? d : d - 1], Val(N))
        # Up to sign: the first nonzero component positive.
        lead = 0
        for x in v
            x != 0 && (lead = x; break)
        end
        key = lead > 0 ? v : map(-, v)
        if haskey(seen, key)
            elements[seen[key]][3][c] = k
        else
            members = Vector{Union{Nothing, NTuple{M, Int}}}(nothing, N)
            members[c] = k
            push!(elements, (c, k, members))
            seen[key] = length(elements)
        end
    end
    return elements
end

function shearlet_layout(sys::BandlimitedShearlets, dims::NTuple{N, Int}) where {N}
    N in (2, 3) || throw(ArgumentError("band-limited shearlets support 2-D and 3-D data, got size $(dims)"))
    layout = [_lowpass_index(Val(N))]
    for j in 1:_nscales(sys, dims)
        for (c, k, _) in _bl_scale_elements(Val(N), _bl_shears(sys, j))
            push!(layout, ShearletIndex{N - 1}(j, c, k))
        end
    end
    return layout
end

shearlet_layout(::BandlimitedShearlets, dims::Tuple) =
    throw(ArgumentError("band-limited shearlets support 2-D and 3-D data, got size $(dims)"))

# Radial windows and lowpass at ω (in the units above).
function _bl_radial(sys::BandlimitedShearlets, ω::NTuple{N, Float64}, j::Int) where {N}
    if sys.smooth
        lo = prod(d -> _meyer_phi(ω[d] / 4^j), 1:N)
        hi = prod(d -> _meyer_phi(ω[d] / 4^(j - 1)), 1:N)
        return sqrt(max(0.0, lo^2 - hi^2))
    end
    return _meyer_psi1(maximum(abs, ω) / 4^(j - 1))
end

_bl_lowpass(sys::BandlimitedShearlets, ω::NTuple{N, Float64}) where {N} =
    sys.smooth ? prod(d -> _meyer_phi(ω[d]), 1:N) : _meyer_phi(maximum(abs, ω))

function _shearlet_masks(sys::BandlimitedShearlets, dims::NTuple{N, Int}, ::Type{Tf}) where {N, Tf}
    layout = shearlet_layout(sys, dims)
    J = _nscales(sys, dims)
    X = 2.0^(2J - 1)
    scale = ntuple(d -> X / (dims[d] / 2), Val(N))
    shears = [_bl_shears(sys, j) for j in 1:J]
    # Per scale, the shears of every element in each cone, in layout order.
    members = [[m for (_, _, m) in _bl_scale_elements(Val(N), shears[j])] for j in 1:J]
    radial = zeros(J)
    function evaluate!(v, f)
        ω = ntuple(d -> f[d] * scale[d], Val(N))
        cone = argmax(abs.(ω))           # the lowest axis on ties
        wc = ω[cone]
        r = ntuple(i -> (e = i < cone ? i : i + 1; wc == 0 ? 0.0 : ω[e] / wc), Val(N - 1))
        v[1] = _bl_lowpass(sys, ω)
        i = 1
        for j in 1:J
            R = _bl_radial(sys, ω, j)
            s = shears[j]
            for m in members[j]
                i += 1
                k = m[cone]
                v[i] = (R == 0 || k === nothing) ? 0.0 : R * prod(e -> _meyer_psi2(s * r[e] + k[e]), 1:(N - 1))
            end
        end
        return v
    end
    return layout, _pointwise_masks(evaluate!, length(layout), dims, Tf)
end
