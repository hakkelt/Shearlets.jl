# One-dimensional Meyer-type windows of the band-limited shearlets.
#
# All are functions of a frequency in the units of the continuous construction
# (Häuser & Steidl, "Fast Finite Shearlet Transform: a tutorial",
# arXiv:1202.1773, §2.1 and §2.4), evaluated in Float64; the masks are converted
# to the filter precision afterwards.

# Meyer's auxiliary function: 0 below 0, 1 above 1, and in between the C³
# polynomial whose derivative is −140 x³ (x − 1)³ (eq. 1).  v(x) + v(1 − x) = 1.
function _meyer_aux(x::Float64)
    x <= 0 && return 0.0
    x >= 1 && return 1.0
    return x^4 * (35 + x * (-84 + x * (70 - 20x)))
end

# b of eq. (2): supported on 1 ≤ |ω| ≤ 4, equal to 1 at |ω| = 2.
function _meyer_b(ω::Float64)
    a = abs(ω)
    1 <= a <= 2 && return sinpi(_meyer_aux(a - 1) / 2)
    2 < a <= 4 && return cospi(_meyer_aux(a / 2 - 1) / 2)
    return 0.0
end

# ψ̂₁ of eq. (4), the radial window: Σⱼ≥₀ ψ̂₁(4⁻ʲω)² = 1 for |ω| ≥ 1.
_meyer_psi1(ω::Float64) = sqrt(_meyer_b(2ω)^2 + _meyer_b(ω)^2)

# ψ̂₂ of eq. (6), the angular bump on [−1, 1]: Σₖ ψ̂₂(ω + k)² = 1 for all ω.
_meyer_psi2(ω::Float64) = sqrt(ω <= 0 ? _meyer_aux(1 + ω) : _meyer_aux(1 - ω))

# ϕ of §2.4, the 1-D lowpass: 1 for |ω| ≤ ½, 0 for |ω| ≥ 1, and
# ϕ(ω)² + ψ̂₁(ω)² = 1 in between (eq. 17).
function _meyer_phi(ω::Float64)
    a = abs(ω)
    a <= 0.5 && return 1.0
    a < 1 && return cospi(_meyer_aux(2a - 1) / 2)
    return 0.0
end
