# The mask engine: one FFT of the input, then per subband a masked product and
# an inverse FFT (analysis); per subband an FFT and a masked accumulation, then
# one inverse FFT (synthesis).
#
# Data are transformed on the `rfft` grid.  Complex data are split into their
# real and imaginary parts, two real channels that go through the same masks, so
# the transform is real-linear bit for bit and a complex input costs two real
# ones.  The spectra of the channels are stacked along a trailing axis.
#
# The subbands are independent, so both passes split them into contiguous chunks,
# one task per FFT buffer (`MultiscaleFilterBanks._foreach_chunk`); the synthesis
# gives each task after the first its own partial sum (`_chunked_sum!`).  FFT
# plans are single-threaded.

"""
    ShearletWorkspace{Td, Tf, N}

Masks, FFT plans and buffers of a shearlet system at one data size and element
type `Td` (filters in `Tf = real(float(Td))`).  Build with
[`make_shearlet_workspace`](@ref) and pass it as `workspace` to
[`shearlet_forward`](@ref) / [`shearlet_inverse`](@ref) to keep repeated calls
out of the allocator.

A workspace holds scratch buffers, so it must not serve two transforms at the
same time; it threads internally instead.
"""
struct ShearletWorkspace{
        Td, Tf, N, S <: ShearletSystem, L, M <: _FrequencyMask{Tf},
        C <: AbstractArray{Complex{Tf}}, CM <: AbstractMatrix{Complex{Tf}}, B <: AbstractArray{Complex{Tf}, N},
        V <: AbstractVector{Complex{Tf}}, R <: AbstractArray{Tf, N}, P, Q,
    }
    system::S
    dims::NTuple{N, Int}
    layout::Vector{L}
    masks::Vector{M}
    X::C                    # channel spectra, `(rfft dims..., channels)`; synthesis accumulator
    Xm::CM                  # the same memory as `(points, channels)`
    partials::Vector{C}     # per task ≥ 2: partial synthesis sum
    partials_m::Vector{CM}
    bufs::Vector{B}         # per task: FFT buffer (consumed by the inverse FFT)
    bufs_v::Vector{V}       # the same memory as vectors (a linear view of a matrix reshapes it)
    rbufs::Vector{Vector{R}} # per task and channel: real-space buffer
    pfwd::P                 # rfft, real data → spectrum
    pinv::Q                 # unnormalised inverse (brfft)
    scale::Tf               # 1 / prod(dims)
end

_nchannels(::Type{<:Real}) = 1
_nchannels(::Type{<:Complex}) = 2
_nchannels(::ShearletWorkspace{Td}) where {Td} = _nchannels(Td)
_ntasks(ws::ShearletWorkspace) = length(ws.bufs)
_threaded(threading::ThreadingPolicy) = !(threading isa Disabled)

"""
    make_shearlet_workspace(T, dims, system; flags = FFTW.MEASURE) -> ShearletWorkspace

Build the masks, buffers and FFT plans of `system` for data of element type `T`
(real or complex) and size `dims`.  `flags` are the FFTW planner flags.
"""
function make_shearlet_workspace(::Type{T}, dims::NTuple{N, Int}, sys::ShearletSystem; flags::UInt32 = FFTW.MEASURE) where {T <: Number, N}
    Td = float(T)
    layout, masks = _shearlet_masks(sys, dims, real(Td))
    nt = max(1, min(Threads.nthreads(), length(masks)))
    return _assemble_workspace(sys, dims, Td, layout, masks, Array{real(Td), N}(undef, ntuple(_ -> 0, N)), nt) do rbuf, buf
        # FFTW's own threads off: the engine threads over subbands instead.
        _with_fftw_threads(1) do
            f = flags | FFTW.UNALIGNED
            plan_rfft(rbuf; flags = f), plan_brfft(buf, dims[1]; flags = f)
        end
    end
end

# A workspace around `masks`, with buffers like `proto` (an array of the filter
# type, host or device) for `nt` tasks and the FFT plans `plans(rbuf, buf)` makes
# for a real buffer and a spectrum buffer.
function _assemble_workspace(
        plans::F, sys::ShearletSystem, dims::NTuple{N, Int}, ::Type{Td}, layout::Vector, masks::Vector,
        proto::AbstractArray{Tf, N}, nt::Int
    ) where {F, N, Td, Tf}
    nch = _nchannels(Td)
    rd = _rfft_dims(dims)
    zeroed(T, sz) = fill!(similar(proto, T, sz), zero(T))
    X = zeroed(Complex{Tf}, (rd..., nch))
    partials = [similar(X) for _ in 2:nt]
    bufs = [zeroed(Complex{Tf}, rd) for _ in 1:nt]
    rbufs = [[zeroed(Tf, dims) for _ in 1:nch] for _ in 1:nt]
    pfwd, pinv = plans(rbufs[1][1], bufs[1])
    Xm = reshape(X, :, nch)
    bufs_v = [vec(b) for b in bufs]
    return ShearletWorkspace{
        Td, Tf, N, typeof(sys), eltype(layout), eltype(masks), typeof(X), typeof(Xm), eltype(bufs),
        eltype(bufs_v), eltype(eltype(rbufs)), typeof(pfwd), typeof(pinv),
    }(
        sys, dims, layout, masks, X, Xm, partials, [reshape(p, :, nch) for p in partials],
        bufs, bufs_v, rbufs, pfwd, pinv, Tf(1 / prod(dims))
    )
end

make_shearlet_workspace(::Type{T}, dims::Tuple, sys::ShearletSystem; kwargs...) where {T <: Number} =
    make_shearlet_workspace(T, Tuple(Int.(dims)), sys; kwargs...)

# The workspace for one call: `workspace` after checking it fits, else a new one.
function _workspace(workspace, sys::ShearletSystem, dims::Tuple, ::Type{Td}) where {Td}
    if workspace === nothing
        return make_shearlet_workspace(Td, dims, sys; flags = FFTW.ESTIMATE)
    end
    workspace.dims == dims ||
        throw(DimensionMismatch("workspace was built for size $(workspace.dims), got $(dims)"))
    _nchannels(workspace) == _nchannels(Td) ||
        throw(ArgumentError("workspace was built for $(eltype(workspace)) data, got $(Td)"))
    workspace.system == sys || throw(ArgumentError("workspace was built for a different shearlet system"))
    return workspace
end

Base.eltype(::ShearletWorkspace{Td}) where {Td} = Td

# Spectrum of channel `c` of `x` into column `c` of `ws.X`, through the task-1
# real buffer unless `x` is already a real array the plan can read.
function _spectrum!(ws::ShearletWorkspace{Td, Tf, N}, x::AbstractArray, c::Int) where {Td, Tf, N}
    dst = selectdim(ws.X, N + 1, c)
    if _plannable(x, Tf, N)
        mul!(dst, ws.pfwd, x)
    else
        rb = ws.rbufs[1][1]
        rb .= _channel.(x, c)
        mul!(dst, ws.pfwd, rb)
    end
    return dst
end

# Whether the FFT plans can read or write `a` in place: a dense, column-major
# array of the filter type (a whole array, or a slice of the stacked subbands).
_plannable(a::AbstractArray, ::Type{Tf}, N::Int) where {Tf} =
    a isa StridedArray{Tf} && ndims(a) == N && strides(a) == Base.size_to_strides(1, size(a)...)

_channel(x::Real, ::Int) = x
_channel(x::Complex, c::Int) = c == 1 ? real(x) : imag(x)

# Analysis: subband i of every channel, handed to `emit(i, t)` once the task-t
# real buffers hold it (`ws.rbufs[t][c]`), unless `target(i, t, c)` names the
# array the inverse FFT writes to directly.
function _engine_forward!(target::F, emit::G, x::AbstractArray, ws::ShearletWorkspace{Td, Tf, N}, threaded::Bool) where {F, G, Td, Tf, N}
    nch = _nchannels(ws)
    for c in 1:nch
        _spectrum!(ws, x, c)
    end
    s = ws.scale
    _foreach_chunk(length(ws.masks), _ntasks(ws), threaded) do range, t
        Z, Zv = ws.bufs[t], ws.bufs_v[t]
        for i in range
            m = ws.masks[i]
            for c in 1:nch
                _mask_apply!(Z, Zv, m, ws.X, ws.Xm, c, s)
                mul!(target(i, t, c), ws.pinv, Z)
            end
            emit(i, t)
        end
    end
    return nothing
end

# Synthesis into `x` from the subbands `source(i)` (arrays of the data size).
function _engine_inverse!(source::S, x::AbstractArray, ws::ShearletWorkspace{Td, Tf, N}, threaded::Bool) where {S, Td, Tf, N}
    nch = _nchannels(ws)
    s = ws.scale
    fill!(ws.X, zero(eltype(ws.X)))
    _chunked_sum!(ws.X, ws.partials, length(ws.masks), threaded) do a, range, t
        am = t == 1 ? ws.Xm : ws.partials_m[t - 1]
        F, Fv = ws.bufs[t], ws.bufs_v[t]
        for i in range
            m = ws.masks[i]
            band = source(i)
            for c in 1:nch
                if nch == 1 && _plannable(band, Tf, N)
                    mul!(F, ws.pfwd, band)
                else
                    rb = ws.rbufs[t][c]
                    rb .= _channel.(band, c)
                    mul!(F, ws.pfwd, rb)
                end
                _mask_accumulate!(a, am, m, F, Fv, c, s)
            end
        end
    end
    if nch == 1 && _plannable(x, Tf, N)
        mul!(x, ws.pinv, selectdim(ws.X, N + 1, 1))
    else
        for c in 1:nch
            mul!(ws.rbufs[1][c], ws.pinv, selectdim(ws.X, N + 1, c))
        end
        _combine!(x, ws.rbufs[1])
    end
    return x
end

# dst = the array of channels `r` (one real channel, or real and imaginary).
_combine!(dst::AbstractArray, r::AbstractVector) = length(r) == 1 ? (dst .= r[1]) : (dst .= complex.(r[1], r[2]))
