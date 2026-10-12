"""
    ShearletSystem

Abstract type of the shearlet systems: a description of the frame that does not
depend on the data size.  The masks themselves are built for one size and
element type by [`make_shearlet_workspace`](@ref).

Subtypes: [`BandlimitedShearlets`](@ref), [`NSSTShearlets`](@ref),
[`CompactShearlets`](@ref).
"""
abstract type ShearletSystem end

"""
    shearlet_layout(system, dims) -> Vector{ShearletIndex}

The subbands of `system` on data of size `dims`, in the order the transforms
stack them: the lowpass first, then the scales from coarse to fine.
"""
function shearlet_layout end

"""
    shearlet_nsubbands(system, dims) -> Int

Number of subbands, lowpass included, that `system` produces on data of size
`dims`: the length of the last axis of the coefficient array.
"""
shearlet_nsubbands(sys::ShearletSystem, dims::Tuple) = length(shearlet_layout(sys, dims))
