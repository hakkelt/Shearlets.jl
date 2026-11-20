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
