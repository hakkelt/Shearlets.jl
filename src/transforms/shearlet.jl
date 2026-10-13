# Public transforms over the mask engine (`engine/workspace.jl`).
#
# Coefficients are one array of size `(size(x)..., nsubbands)`: every subband has
# the size of the input, so stacking them costs no memory over separate arrays
# and keeps the whole set in one allocation, sliced with `selectdim(c, N + 1, i)`.
# `shearlet_layout(system, size(x))[i]` says what subband `i` is.

_coefficient_size(ws::ShearletWorkspace) = (ws.dims..., _nsubbands(ws))

# Masks are built on the host; device data needs a workspace adapted to it.
function _check_host(workspace, x::AbstractArray)
    workspace === nothing && !(x isa Array) &&
        throw(ArgumentError("pass a `workspace` for $(typeof(x)) data: `Adapt.adapt(ArrayType, make_shearlet_workspace(T, dims, system))`"))
    return nothing
end

function _check_coefficients(coeffs::AbstractArray, ws::ShearletWorkspace)
    size(coeffs) == _coefficient_size(ws) ||
        throw(DimensionMismatch("coefficients of size $(size(coeffs)) do not fit the system's $(_coefficient_size(ws))"))
    return nothing
end

"""
    shearlet_forward(x, system::ShearletSystem; workspace = nothing, threading = Auto()) -> Array

Shearlet transform of `x`: an array of size `(size(x)..., n)` whose slice `i`
along the last axis is the subband `shearlet_layout(system, size(x))[i]` (the
lowpass first).  The transform is invariant under circular shifts of `x`;
real input gives real subbands.  The subbands are computed in parallel unless
`threading = Disabled()`.

# Examples
```jldoctest
julia> using Shearlets, Random

julia> x = randn(Xoshiro(1), 64, 64);

julia> c = shearlet_forward(x, BandlimitedShearlets(scales = 2));

julia> size(c)
(64, 64, 13)

julia> sum(abs2, c) ≈ sum(abs2, x)   # a Parseval frame
true
```
"""
function shearlet_forward(
        x::AbstractArray{<:Number}, sys::ShearletSystem;
        workspace::Union{ShearletWorkspace, Nothing} = nothing, threading::ThreadingPolicy = Auto()
    )
    _check_host(workspace, x)
    Td = _data_eltype(x)
    ws = _workspace(workspace, sys, size(x), Td, _threaded(threading))
    coeffs = similar(x, Td, _coefficient_size(ws))
    return shearlet_forward!(coeffs, x, sys; workspace = ws, threading = threading)
end

"""
    shearlet_forward!(coeffs, x, system::ShearletSystem; workspace = nothing, threading = Auto()) -> coeffs

In-place [`shearlet_forward`](@ref).  With a `workspace` it allocates nothing
but the threads' tasks.
"""
function shearlet_forward!(
        coeffs::AbstractArray{<:Number}, x::AbstractArray{<:Number}, sys::ShearletSystem;
        workspace::Union{ShearletWorkspace, Nothing} = nothing, threading::ThreadingPolicy = Auto()
    )
    _check_host(workspace, x)
    threaded = _threaded(threading)
    ws = _workspace(workspace, sys, size(x), _data_eltype(x), threaded)
    _check_coefficients(coeffs, ws)
    _forward_into!(coeffs, x, ws, threaded)
    return coeffs
end

# Subband `i` of a stacked coefficient array.
_band(coeffs::AbstractArray, i::Int) = selectdim(coeffs, ndims(coeffs), i)

function _forward_into!(coeffs::AbstractArray, x::AbstractArray, ws::ShearletWorkspace{Td, Tf, N}, threaded::Bool) where {Td, Tf, N}
    # Every slice is aligned if the first two are: they are equally spaced.
    if _nchannels(ws) == 1 && all(i -> _plannable(_band(coeffs, i), Tf, N), 1:min(2, size(coeffs, N + 1)))
        # Real data into dense slices: the inverse FFT writes the subbands.
        _engine_forward!((i, t, c) -> _band(coeffs, i), (i, t) -> nothing, x, ws, threaded)
    else
        emit(i, t) = (_combine!(_band(coeffs, i), ws.rbufs[t]); nothing)
        _engine_forward!((i, t, c) -> ws.rbufs[t][c], emit, x, ws, threaded)
    end
    return coeffs
end

"""
    shearlet_forward(f, x, system::ShearletSystem; workspace = nothing)

Streaming shearlet transform: calls `f(i, band)` for every subband `i` in
`shearlet_layout` order, reusing one buffer between calls, so the coefficient
set is never held at once.  `band` is valid only until `f` returns — copy or
reduce it there.  The calls come one at a time, from the calling task.
"""
function shearlet_forward(
        f, x::AbstractArray{<:Number}, sys::ShearletSystem;
        workspace::Union{ShearletWorkspace, Nothing} = nothing
    )
    _check_host(workspace, x)
    Td = _data_eltype(x)
    ws = _workspace(workspace, sys, size(x), Td, false)
    out = Td <: Real ? nothing : similar(x, Td)
    target(i, t, c) = ws.rbufs[t][c]
    emit(i, t) = (f(i, out === nothing ? ws.rbufs[t][1] : _combine!(out, ws.rbufs[t])); nothing)
    _engine_forward!(target, emit, x, ws, false)
    return nothing
end

"""
    shearlet_inverse(coeffs, system::ShearletSystem; workspace = nothing, threading = Auto()) -> Array

Invert [`shearlet_forward`](@ref).  For a Parseval system such as
[`BandlimitedShearlets`](@ref) this is also the adjoint of the forward
transform.
"""
function shearlet_inverse(
        coeffs::AbstractArray{<:Number}, sys::ShearletSystem;
        workspace::Union{ShearletWorkspace, Nothing} = nothing, threading::ThreadingPolicy = Auto()
    )
    N = ndims(coeffs) - 1
    x = similar(coeffs, _data_eltype(coeffs), ntuple(d -> size(coeffs, d), N))
    return shearlet_inverse!(x, coeffs, sys; workspace = workspace, threading = threading)
end

"""
    shearlet_inverse!(x, coeffs, system::ShearletSystem; workspace = nothing, threading = Auto()) -> x

In-place [`shearlet_inverse`](@ref).
"""
function shearlet_inverse!(
        x::AbstractArray{<:Number}, coeffs::AbstractArray{<:Number}, sys::ShearletSystem;
        workspace::Union{ShearletWorkspace, Nothing} = nothing, threading::ThreadingPolicy = Auto()
    )
    _check_host(workspace, x)
    threaded = _threaded(threading)
    ws = _workspace(workspace, sys, size(x), promote_type(_data_eltype(x), _data_eltype(coeffs)), threaded)
    _check_coefficients(coeffs, ws)
    return _engine_inverse!(i -> _band(coeffs, i), x, ws, threaded)
end

"""
    shearlet_inverse(f, system::ShearletSystem, dims; T = Float64, workspace = nothing) -> Array

Streaming inverse: `f(i)` must return subband `i` (in `shearlet_layout` order)
as an array of size `dims`.  The subbands are consumed one at a time, from the
calling task.
"""
function shearlet_inverse(
        f, sys::ShearletSystem, dims::Tuple; T::Type{<:Number} = Float64,
        workspace::Union{ShearletWorkspace, Nothing} = nothing
    )
    Td = float(T)
    ws = _workspace(workspace, sys, Tuple(dims), Td, false)
    x = Array{Td}(undef, dims...)
    return _engine_inverse!(f, x, ws, false)
end

"""
    shearlet_spectra(system, dims; T = Float64, synthesis = false) -> Array{T}

The frequency responses of all subbands of `system` on the full, unshifted DFT
grid of a `dims` array: an array of size `(dims..., n)`, in `shearlet_layout`
order.  Subband `i` of `x` is `real(ifft(spectra[.., i] .* fft(x)))`.

With `synthesis = true`, the masks the inverse applies instead: `x` is
`real(ifft(sum(spectra .* fft(subbands))))`.  For a Parseval system the two
are equal.
"""
function shearlet_spectra(sys::ShearletSystem, dims::Tuple; T::Type{<:AbstractFloat} = Float64, synthesis::Bool = false)
    dims = Tuple(Int.(dims))
    N = length(dims)
    ms = _shearlet_masks(sys, dims, T)
    rd = _rfft_dims(dims)
    out = zeros(T, dims..., length(ms.ana))
    for (g, grp) in enumerate(ms.groups), i in grp
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
