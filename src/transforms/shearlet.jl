# The shearlet systems as AbstractMultiscaleTransforms transforms, over the mask
# engine (`engine/workspace.jl`).
#
# Coefficients are one array of size `(size(x)..., nsubbands)`: every subband has
# the size of the input, so stacking them costs no memory over separate arrays
# and keeps the whole set in one allocation, sliced with `selectdim(c, N + 1, i)`.
# `subband_layout(system, size(x))[i]` says what subband `i` is.

_coefficient_size(ws::ShearletWorkspace) = (ws.dims..., _nsubbands(ws))

# Masks are built on the host; device data needs a workspace for its device.
function _check_host(workspace, x::AbstractArray)
    workspace === nothing && !_on_host(x) &&
        throw(ArgumentError("pass a `workspace` for $(typeof(x)) data: `make_workspace(system, x)`"))
    return nothing
end

# Whether the data of `x` lives on the host: any array but a device array (or a
# wrapper of one), BitArrays, ranges and sparse arrays included.
_on_host(x::AbstractArray) = !is_device_array(x)

function _check_coefficients(coeffs::AbstractArray, ws::ShearletWorkspace)
    size(coeffs) == _coefficient_size(ws) ||
        throw(DimensionMismatch("coefficients of size $(size(coeffs)) do not fit the system's $(_coefficient_size(ws))"))
    return nothing
end

AbstractMultiscaleTransforms.supports_streaming(::ShearletSystem) = true

"""
    subband(system::ShearletSystem, c, i) -> AbstractArray

Subband `i` of the coefficients `c` of `system`: the view `selectdim(c, ndims(c), i)`.
"""
AbstractMultiscaleTransforms.subband(::ShearletSystem, c::AbstractArray, i::Integer) = selectdim(c, ndims(c), i)

"""
    similar_coefficients(system::ShearletSystem, x) -> AbstractArray

Uninitialized coefficients of `system` for data like `x`: an array like `x`
(host or device) of element type `float(eltype(x))` and size
`(size(x)..., nsubbands(system, size(x)))`.  `x` must be indexed from 1.
"""
function AbstractMultiscaleTransforms.similar_coefficients(sys::ShearletSystem, x::AbstractArray)
    Base.require_one_based_indexing(x)
    return similar(x, data_eltype(x), (size(x)..., nsubbands(sys, size(x))))
end

AbstractMultiscaleTransforms.signal_size(::ShearletSystem, c::AbstractArray) = Base.front(size(c))

"""
    analyze(system::ShearletSystem, x; workspace = nothing, threaded = nothing) -> AbstractArray

Shearlet transform of `x`: an array of size `(size(x)..., n)` whose slice `i`
along the last axis (`subband(system, c, i)`) is the subband
`subband_layout(system, size(x))[i]` (the lowpass first).  The transform is
invariant under circular shifts of `x`; real input gives real subbands, and
complex input is transformed as its real and imaginary parts, so
`analyze(system, complex.(a, b)) == complex.(analyze(system, a), analyze(system, b))`
exactly.  The subbands are computed in parallel unless `threaded = false`
(`nothing` threads).  The data, and every coefficient and output array of the
transforms below, must be indexed from 1 (an `ArgumentError` otherwise).

`workspace` is a [`ShearletWorkspace`](@ref) for this system, element type and
size; device data needs one (see
[`make_workspace`](@ref make_workspace(::ShearletSystem, ::AbstractArray))).
Without one, masks and plans are built for the call.

# Examples
```jldoctest
julia> using Shearlets, Random

julia> x = randn(Xoshiro(1), 64, 64);

julia> c = analyze(BandlimitedShearlets(scales = 2), x);

julia> size(c)
(64, 64, 13)

julia> sum(abs2, c) ≈ sum(abs2, x)   # a Parseval frame
true
```
"""
function AbstractMultiscaleTransforms.analyze(
        sys::ShearletSystem, x::AbstractArray{<:Number};
        workspace::Union{ShearletWorkspace, Nothing} = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    Base.require_one_based_indexing(x)
    _check_host(workspace, x)
    Td = data_eltype(x)
    ws = _workspace(workspace, sys, size(x), Td, _threaded(threaded))
    coeffs = similar(x, Td, _coefficient_size(ws))
    return analyze!(coeffs, sys, x; workspace = ws, threaded)
end

"""
    analyze!(c, system::ShearletSystem, x; workspace = nothing, threaded = nothing) -> c

In-place [`analyze`](@ref analyze(::ShearletSystem, ::AbstractArray{<:Number})): writes the
coefficients of `x` into `c`, which must have size `(size(x)..., n)` and a
complex element type when `x` is complex (`DimensionMismatch` and
`ArgumentError` otherwise).  With a `workspace` it allocates nothing but the
threads' tasks.
"""
function AbstractMultiscaleTransforms.analyze!(
        coeffs::AbstractArray{<:Number}, sys::ShearletSystem, x::AbstractArray{<:Number};
        workspace::Union{ShearletWorkspace, Nothing} = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    Base.require_one_based_indexing(coeffs, x)
    _check_host(workspace, x)
    data_eltype(coeffs) <: Real && data_eltype(x) <: Complex &&
        throw(ArgumentError("complex data need complex coefficients, got $(eltype(coeffs))"))
    use_threads = _threaded(threaded)
    ws = _workspace(workspace, sys, size(x), data_eltype(x), use_threads)
    _check_coefficients(coeffs, ws)
    _forward_into!(coeffs, x, ws, use_threads)
    return coeffs
end

# The subband of an engine output index; the engine has no lowpass of its own
# (the lowpass is subband 1).
_band_index(ix::Tuple{Int}) = ix[1]
_band_index(::Nothing) = throw(ArgumentError("a shearlet engine has no separate lowpass"))

# Subband `i` (or the subbands in a range) of a stacked coefficient array.
_band(coeffs::AbstractArray, i::Union{Int, UnitRange{Int}}) = selectdim(coeffs, ndims(coeffs), i)

# Subbands the plans cannot take are written (read) one at a time through
# `ws.bbuf`, and an output they cannot take is staged in `ws.xbuf`.
function _forward_into!(coeffs::AbstractArray, x::AbstractArray, ws::ShearletWorkspace, threaded::Bool)
    xin = _engine_input(ws, x)
    if _direct_bands(ws, coeffs)
        mask_bank_forward!(_Stack(coeffs), xin, ws.engine; threaded)
    else
        mask_bank_stream_forward!((ix, band) -> (copyto!(_band(coeffs, _band_index(ix)), band); nothing), ws.bbuf, xin, ws.engine)
    end
    return coeffs
end

function _inverse_from!(x::AbstractArray, coeffs::AbstractArray, ws::ShearletWorkspace, threaded::Bool)
    out = _direct(ws, x) ? x : ws.xbuf
    if _direct_bands(ws, coeffs)
        mask_bank_inverse!(out, _Stack(coeffs), ws.engine; threaded)
    else
        mask_bank_stream_inverse!(ix -> copyto!(ws.bbuf, _band(coeffs, _band_index(ix))), out, ws.engine)
    end
    out === x || copyto!(x, out)
    return x
end

"""
    analyze(f, system::ShearletSystem, x; workspace = nothing, threaded = nothing) -> nothing

Streaming shearlet transform: calls `f(i, band)` for every subband `i` in
`subband_layout` order, reusing one buffer between calls, so the coefficient
set is never held at once.  `band` is valid only until `f` returns — copy or
reduce it there.  The subbands are computed one at a time: `f` is called once
per subband, in that order, from the calling task.  `threaded` has no effect
(it is accepted for the common interface).
"""
function AbstractMultiscaleTransforms.analyze(
        f, sys::ShearletSystem, x::AbstractArray{<:Number};
        workspace::Union{ShearletWorkspace, Nothing} = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    Base.require_one_based_indexing(x)
    _check_host(workspace, x)
    ws = _workspace(workspace, sys, size(x), data_eltype(x), false)
    mask_bank_stream_forward!((ix, band) -> (f(_band_index(ix), band); nothing), ws.bbuf, _engine_input(ws, x), ws.engine)
    return nothing
end

"""
    synthesize(system::ShearletSystem, c; workspace = nothing, threaded = nothing) -> AbstractArray

Invert [`analyze`](@ref analyze(::ShearletSystem, ::AbstractArray{<:Number})): the array of
size `size(c)[1:end-1]` and element type `float(eltype(c))` whose transform is
`c`.  For a Parseval system such as [`BandlimitedShearlets`](@ref) this is
also the adjoint of the forward transform; the others are dual frames.  Complex
coefficients give a complex result.  `workspace` and `threaded` are those of
[`analyze`](@ref analyze(::ShearletSystem, ::AbstractArray{<:Number})).
"""
function AbstractMultiscaleTransforms.synthesize(
        sys::ShearletSystem, coeffs::AbstractArray{<:Number};
        workspace::Union{ShearletWorkspace, Nothing} = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    Base.require_one_based_indexing(coeffs)
    x = similar(coeffs, data_eltype(coeffs), signal_size(sys, coeffs))
    return synthesize!(x, sys, coeffs; workspace, threaded)
end

"""
    synthesize!(x, system::ShearletSystem, c; workspace = nothing, threaded = nothing) -> x

In-place [`synthesize`](@ref synthesize(::ShearletSystem, ::AbstractArray{<:Number})): writes
the inverse of `c` into `x`, of size `size(c)[1:end-1]` and a complex element
type when `c` is complex (`DimensionMismatch` and `ArgumentError` otherwise).
With a `workspace` and `threaded = false` it allocates nothing.
"""
function AbstractMultiscaleTransforms.synthesize!(
        x::AbstractArray{<:Number}, sys::ShearletSystem, coeffs::AbstractArray{<:Number};
        workspace::Union{ShearletWorkspace, Nothing} = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    Base.require_one_based_indexing(x, coeffs)
    _check_host(workspace, x)
    _check_host(workspace, coeffs)
    data_eltype(x) <: Real && data_eltype(coeffs) <: Complex &&
        throw(ArgumentError("complex coefficients need a complex output, got $(eltype(x))"))
    use_threads = _threaded(threaded)
    ws = _workspace(workspace, sys, size(x), data_eltype(x), use_threads)
    _check_coefficients(coeffs, ws)
    return _inverse_from!(x, coeffs, ws, use_threads)
end

"""
    synthesize!(f, x, system::ShearletSystem; workspace = nothing, threaded = nothing) -> x

Streaming inverse into `x`: `f(i)` must return subband `i` (in
`subband_layout` order) as an array of size `size(x)`.  The subbands are
consumed one at a time: `f` is called once per subband, in `subband_layout`
order, from the calling task, and each result is read before the next call, so
`f` may reuse one buffer.  `threaded` has no effect (it is accepted for the
common interface).
"""
function AbstractMultiscaleTransforms.synthesize!(
        f, x::AbstractArray{<:Number}, sys::ShearletSystem;
        workspace::Union{ShearletWorkspace, Nothing} = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    Base.require_one_based_indexing(x)
    _check_host(workspace, x)
    ws = _workspace(workspace, sys, size(x), data_eltype(x), false)
    out = _direct(ws, x) ? x : ws.xbuf
    mask_bank_stream_inverse!(out, ws.engine) do ix
        i = _band_index(ix)
        band = f(i)
        Base.require_one_based_indexing(band)
        size(band) == ws.dims ||
            throw(DimensionMismatch("f($(i)) returned a band of size $(size(band)), expected $(ws.dims)"))
        _direct(ws, band) ? band : copyto!(ws.bbuf, band)
    end
    out === x || copyto!(x, out)
    return x
end

"""
    shearlet_spectra(system, [T = Float64,] dims; synthesis = false) -> Array{T}

The frequency responses of all subbands of `system`, of the real element type
`T`, on the full, unshifted DFT grid of a `dims` array: an array of size
`(dims..., n)`, in `subband_layout` order.  Subband `i` of `x` is `real(ifft(spectra[.., i] .* fft(x)))`.

With `synthesis = true`, the masks the inverse applies instead: `x` is
`real(ifft(sum(spectra .* fft(subbands))))`.  For a Parseval system the two
are equal.  `dims` has 2 or 3 entries (2 for `NSSTShearlets`).

# Examples
```jldoctest
julia> using Shearlets

julia> spectra = shearlet_spectra(BandlimitedShearlets(scales = 1), (32, 32));

julia> size(spectra)
(32, 32, 5)

julia> sum(abs2, spectra; dims = 3) ≈ ones(32, 32)   # a Parseval frame
true
```
"""
shearlet_spectra(sys::ShearletSystem, dims::Tuple; synthesis::Bool = false) = shearlet_spectra(sys, Float64, dims; synthesis)

function shearlet_spectra(sys::ShearletSystem, ::Type{T}, dims::Tuple; synthesis::Bool = false) where {T <: AbstractFloat}
    dims = Tuple(Int.(dims))
    N = length(dims)
    ms = _system_masks(sys, dims, T)
    rd = _rfft_dims(dims)
    out = zeros(T, dims..., length(ms.ana))
    for (g, grp) in enumerate(_groups(ms)), i in grp
        half = synthesis ? _dense_mask(ms.syn[g], rd) : _dense_mask(ms.ana[i], rd)
        synthesis && ms.post !== nothing && (half .*= ms.post)
        dst = selectdim(out, N + 1, i)
        for I in CartesianIndices(dims)
            # The other half of the first axis is the mirror image M(-f) = M(f).
            if I[1] <= rd[1]
                dst[I] = half[I]
            else
                J = ntuple(d -> d == 1 ? dims[1] - I[1] + 2 : mod(dims[d] - I[d] + 1, dims[d]) + 1, N)
                dst[I] = half[J...]
            end
        end
    end
    return out
end
