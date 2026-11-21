# The workspace: a shearlet system's masks run by MultiscaleFilterBanks'
# `MaskBankEngine`, one FFT of the input, then per subband a masked product and
# an inverse FFT (analysis); per group of subbands their sum, an FFT and the
# group's mask, then the post-factor and one inverse FFT (synthesis).
#
# Data are transformed on the `rfft` grid.  Complex data are split into their
# real and imaginary parts, two real channels that go through the same masks
# (the engine's `split_complex`), so the transform is real-linear bit for bit
# and a complex input costs two real ones.  On the host the engine threads over
# the subbands with single-threaded FFT plans; on a device it transforms them in
# chunks, one batched FFT per chunk (the Adapt extension).
#
# The engine's FFT plans read and write the caller's real arrays directly: the
# input, the output and the subbands, which are the slices of the stacked
# coefficients.  Real arrays the plans cannot take (views with gaps, other
# element types) go through the workspace's buffers.

"""
    ShearletWorkspace{Td, Tf, N} <: AbstractTransformWorkspace

Masks, FFT plans and buffers of a shearlet system at one data size and element
type `Td` (filters in `Tf = real(float(Td))`).  Build with
[`make_workspace`](@ref make_workspace(::ShearletSystem, ::Type{<:Number}, ::Dims))
and pass it as `workspace` to [`analyze`](@ref analyze(::ShearletSystem, ::AbstractArray{<:Number}))
/ [`synthesize`](@ref synthesize(::ShearletSystem, ::AbstractArray{<:Number})) to keep
repeated calls out of the allocator, or call the verbs on the workspace alone
(`analyze(ws, x)`, `synthesize!(x, ws, c)`, ...).  `transform(ws)` is the
system, `eltype(ws)` the data element type `Td` and `size(ws)` the data size.

A workspace holds scratch buffers, so it must not serve two transforms at the
same time; it threads internally instead.
"""
struct ShearletWorkspace{Td, Tf, N, S <: ShearletSystem, MS <: _MaskSet, E <: MaskBankEngine, B <: AbstractArray{Td, N}} <: AbstractTransformWorkspace
    system::S
    dims::NTuple{N, Int}
    masks::MS               # the system's masks, on the host
    engine::E
    aligned::Bool           # whether the plans need 16-byte aligned arrays
    xbuf::B                 # the data, for arrays the plans cannot take
    bbuf::B                 # one subband: streaming, and subbands the plans cannot take
end

function ShearletWorkspace{Td}(sys, dims::NTuple{N, Int}, masks, engine, aligned::Bool, xbuf, bbuf) where {Td, N}
    Tf = real(Td)
    return ShearletWorkspace{Td, Tf, N, typeof(sys), typeof(masks), typeof(engine), typeof(xbuf)}(
        sys, dims, masks, engine, aligned, xbuf, bbuf
    )
end

_with_engine(ws::ShearletWorkspace, engine) =
    ShearletWorkspace{eltype(ws)}(ws.system, ws.dims, ws.masks, engine, ws.aligned, ws.xbuf, ws.bbuf)

_nsubbands(ws::ShearletWorkspace) = length(ws.engine.items)
# Whether a call threads: asked to, and not inside a caller's NestedThreading budget.
_threaded(threaded::Union{Bool, Nothing}) = loop_threaded(threaded !== false)

"""
    make_workspace(system::ShearletSystem, T::Type{<:Number}, dims::Dims; flags = FFTW.MEASURE) -> ShearletWorkspace

Build the masks, buffers and FFT plans of `system` for host data of element
type `T` (real or complex; integers count as `float(T)`) and size `dims`.
`flags` are the FFTW planner flags.  The workspace threads over the subbands.
"""
function MultiscaleTransformsBase.make_workspace(sys::ShearletSystem, ::Type{T}, dims::Dims; flags::UInt32 = FFTW.MEASURE) where {T <: Number}
    return _make_workspace(T, dims, sys, flags, true)
end

"""
    make_workspace(system::ShearletSystem, x::AbstractArray; flags = FFTW.MEASURE, tune = false) -> ShearletWorkspace

The workspace of `system` for data like `x`: its element type, size and
device.  For a host array this is `make_workspace(system, eltype(x), size(x); flags)`.

For a device array (a `CuArray`, say) the masks are built on the host and
moved to the device by the Adapt extension; the FFT plans are the device FFT
library's.  A device workspace transforms its subbands in chunks, one batched
FFT per chunk, with a chunk size that by default follows from the grid size.
`tune = true` measures instead (`MultiscaleFilterBanks.mask_bank_tune`): it
times one subband at a time and chunks of powers of two subbands, each with the
synthesis chunked or not, on a forward and an inverse transform, and keeps the
default unless another is 5% faster.  Tuning holds a coefficient array and the
stacks of one more chunk size on the device meanwhile; the transform itself is
unchanged.  On an A100 (Float32 and ComplexF32) it pays off only for the
`NSSTShearlets`: at 256² and 512² it adds 40–280 ms to the construction and
makes the analysis 10–45% and the synthesis 1.2–3.4× faster, which repays
the construction after about 230–660 analysis–synthesis pairs.  For the
`BandlimitedShearlets` and `CompactShearlets` (2-D and 3-D) the default
chunking already measures best, and tuning only adds 0.1–4.4 s to the
construction.  On the host `tune` does nothing: a host workspace threads
over subbands instead.

```julia
ws = make_workspace(system, CuArray{Float32}(undef, 256, 256); tune = true)
```
"""
function MultiscaleTransformsBase.make_workspace(sys::ShearletSystem, x::AbstractArray; flags::UInt32 = FFTW.MEASURE, tune::Bool = false)
    ws = make_workspace(sys, eltype(x), size(x); flags)
    _on_host(x) && return ws
    dws = _to_device(ws, x)
    return tune ? _tune(dws) : dws
end

# `ws` moved to the device of `x`; a method of the Adapt extension.
_to_device(ws, x) =
    throw(ArgumentError("a workspace for $(typeof(x)) data needs the Adapt extension: load Adapt"))

# A host workspace, threading over the subbands when `threaded`.
function _make_workspace(::Type{T}, dims::NTuple{N, Int}, sys::ShearletSystem, flags::UInt32, threaded::Bool) where {T, N}
    Td = float(T)
    Tf = real(Td)
    masks = _system_masks(sys, dims, Tf)
    # The plans run on the slices of stacked coefficients too, which are 16-byte
    # aligned only when a slice is a multiple of 16 bytes; else they are planned
    # for any alignment (`UNALIGNED` is a little slower).
    aligned = prod(dims) * sizeof(Tf) % 16 == 0
    engine = _engine(masks, Td, zeros(Tf, dims), _host_plans(flags, aligned); tasks = threaded ? nothing : 1)
    return ShearletWorkspace{Td}(sys, dims, masks, engine, aligned, zeros(Td, dims), zeros(Td, dims))
end

# The host FFT plans of an engine (`plans` of `_engine`): single-threaded, since
# the engine threads over subbands instead, and for any alignment unless `aligned`.
function _host_plans(flags::UInt32, aligned::Bool)
    fl = aligned ? flags : flags | FFTW.UNALIGNED
    return function (rb, X)
        return with_fftw_threads(1) do
            plan_rfft(rb; flags = fl), plan_brfft(X, size(rb, 1); flags = fl)
        end
    end
end

"""
    _engine(masks, Td, rb, plans; tasks, planner, forward_planner, chunk_samples) -> MaskBankEngine

The engine of `masks` for data of element type `Td` and the size of `rb`, a
real array of the filter type (host or device) that `plans(rb, X)` turns into the
pair (`rfft` plan, unnormalised inverse plan) with `X` a spectrum buffer.  The
other keywords are those of `MaskBankEngine` (`planner` and `forward_planner`:
chunked paths, for devices).

A system with one subband per group and the analysis masks for synthesis is one
flat bank; otherwise the groups' synthesis masks apply to the sums of their
subbands, which the engine forms in space (`IdentityMask`), one FFT per group.
"""
function _engine(
        masks::_MaskSet, ::Type{Td}, rb::AbstractArray{Tf, N}, plans; tasks = nothing,
        planner = nothing, forward_planner = nothing, chunk_samples::Int = DEFAULT_CHUNK_SAMPLES
    ) where {Td, Tf, N}
    dims = size(rb)
    X = _zeros_like(rb, Complex{Tf}, _rfft_dims(dims))
    pf, pb = plans(rb, X)
    s = Tf(1 / prod(dims))
    bank, items, groups = _bank(masks, masks.groups)
    output = planner === nothing ? nothing : _zeros_like(rb, Td, dims)
    return MaskBankEngine(
        [bank], items, X; pfwd = pf, pinv = s * pb, groups, post = masks.post, pbwd = pb, scale = s,
        split_complex = Td <: Complex, planner, forward_planner, output, chunk_samples, tasks
    )
end

# The bank, items and group synthesis masks of the engine: one flat bank, or
# groups whose synthesis masks apply to the sums of their subbands.
_bank(masks::_MaskSet, ::Nothing) = (MaskBank(masks.ana), [MaskItem((i,), 0, 1, i) for i in eachindex(masks.ana)], nothing)

function _bank(masks::_MaskSet, groups::Vector{UnitRange{Int}})
    items = [MaskItem((i,), g, 1, i) for (g, grp) in enumerate(groups) for i in grp]
    return MaskBank(masks.ana, fill(IdentityMask(), length(masks.ana))), items, masks.syn
end

# A zeroed array like `proto` (host or device).
_zeros_like(proto::AbstractArray, ::Type{T}, sz::Tuple) where {T} = fill!(similar(proto, T, sz), zero(T))

# The masks of `system` at size `dims`, after `subband_layout` checked that the
# system supports that many dimensions.
function _system_masks(sys::ShearletSystem, dims::Tuple, ::Type{Tf}) where {Tf}
    n = length(subband_layout(sys, dims))
    masks = _shearlet_masks(sys, dims, Tf)
    @assert length(masks.ana) == n
    return masks
end

# The workspace for one call: `workspace` after checking it fits, else a new one.
function _workspace(workspace, sys::ShearletSystem, dims::Tuple, ::Type{Td}, threaded::Bool) where {Td}
    if workspace === nothing
        return _make_workspace(Td, Tuple(Int.(dims)), sys, FFTW.ESTIMATE, threaded)
    end
    return check_workspace(workspace, sys, Td, dims)
end

MultiscaleTransformsBase.transform(ws::ShearletWorkspace) = ws.system
Base.eltype(::Type{<:ShearletWorkspace{Td}}) where {Td} = Td
Base.size(ws::ShearletWorkspace) = ws.dims

# The stacked coefficients as the engine's container: subband i is slice i.
struct _Stack{A <: AbstractArray}
    coeffs::A
end

MultiscaleFilterBanks.mask_bank_subband(s::_Stack, (i,)::Tuple{Int}) = selectdim(s.coeffs, ndims(s.coeffs), i)
MultiscaleFilterBanks.mask_bank_storage(s::_Stack) = s.coeffs

# Whether the FFT plans of a real workspace can read or write `a` in place: a
# dense, column-major array of the filter type and the data size, aligned like
# the buffers the plans were made for (unless they take any alignment).  A
# complex workspace reads and writes its arrays by broadcasts instead.
_plannable(ws::ShearletWorkspace{Td, Tf, N}, a::AbstractArray) where {Td, Tf, N} =
    a isa StridedArray{Tf} && ndims(a) == N && size(a) == ws.dims &&
    strides(a) == Base.size_to_strides(1, size(a)...) && (!ws.aligned || _fftw_aligned(a))

_fftw_aligned(a::AbstractArray) = parent(a) === a ? !(a isa Array) || FFTW.alignment_of(a) == 0 : _fftw_aligned_in(a, parent(a))
_fftw_aligned_in(a, ::Array) = FFTW.alignment_of(a) == 0
_fftw_aligned_in(a, p) = _fftw_aligned(p)   # device arrays: their FFT library needs no alignment

# Whether the plans take every subband of the stacked coefficients: every slice
# is, if the first two are (they are equally spaced).
_plannable_bands(ws::ShearletWorkspace, coeffs::AbstractArray) =
    all(i -> _plannable(ws, selectdim(coeffs, ndims(coeffs), i)), 1:min(2, size(coeffs, ndims(coeffs))))

# `x` as the engine reads it: complex data through broadcasts, real data
# through the plans, staged in `ws.xbuf` when they cannot take it.
_engine_input(ws::ShearletWorkspace{<:Complex}, x::AbstractArray) = x
_engine_input(ws::ShearletWorkspace{<:Real}, x::AbstractArray) = _plannable(ws, x) ? x : copyto!(ws.xbuf, x)

# Whether the engine writes the output `x` (or the subbands `coeffs`) itself.
_direct(ws::ShearletWorkspace{<:Complex}, a::AbstractArray) = true
_direct(ws::ShearletWorkspace{<:Real}, a::AbstractArray) = _plannable(ws, a)
_direct_bands(ws::ShearletWorkspace{<:Complex}, coeffs::AbstractArray) = true
_direct_bands(ws::ShearletWorkspace{<:Real}, coeffs::AbstractArray) = _plannable_bands(ws, coeffs)

# A device workspace with the chunk length of its engine measured fastest
# (`MultiscaleFilterBanks.mask_bank_tune`), on zeros.
function _tune(ws::ShearletWorkspace{Td}) where {Td}
    x = _zeros_like(ws.xbuf, Td, ws.dims)
    c = _zeros_like(ws.xbuf, Td, (ws.dims..., _nsubbands(ws)))
    engine = mask_bank_tune(ws.engine, _Stack(c), x)
    finalize(x)
    finalize(c)
    return _with_engine(ws, engine)
end
