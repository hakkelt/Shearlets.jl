"""
    ShearletSystem <: AbstractMultiscaleTransform

Abstract type of the shearlet systems: a description of the frame that does not
depend on the data size.  The masks themselves are built for one size and
element type by [`make_workspace`](@ref make_workspace(::ShearletSystem, ::Type{<:Number}, ::Dims)).

Subtypes: [`BandlimitedShearlets`](@ref), [`NSSTShearlets`](@ref),
[`CompactShearlets`](@ref).
"""
abstract type ShearletSystem <: AbstractMultiscaleTransform end

"""
    subband_layout(system::ShearletSystem, dims) -> Vector{ShearletIndex}

The subbands of `system` on data of size `dims`, in the order the transforms
stack them: the lowpass first, then the scales from coarse to fine.
`nsubbands(system, dims)` is its length, the length of the last axis of the
coefficient array.
"""
subband_layout(::ShearletSystem, ::Tuple)

# The systems print as the keyword call that rebuilds them.  Filter taps are
# printed in full up to this many, so that the printed form parses back; larger
# arrays as their size and element type, with the taps in the `text/plain`
# display.
const _SHOW_MAX_TAPS = 32

_show_in_full(a::AbstractArray) = length(a) <= _SHOW_MAX_TAPS

function _show_taps(io::IO, a::AbstractArray)
    if _show_in_full(a)
        show(io, collect(a))
    else
        print(io, "<")
        summary(io, collect(a))
        print(io, ">")
    end
    return nothing
end

# The compact form, then the taps of each array of `named` (name => taps, the
# arrays the compact form prints) that it summarizes.
function _show_with_taps(io::IO, sys::ShearletSystem, named)
    show(io, sys)
    for (name, a) in named
        _show_in_full(a) && continue
        print(io, "\n  ", name, " = ")
        show(io, MIME("text/plain"), collect(a))
    end
    return nothing
end
