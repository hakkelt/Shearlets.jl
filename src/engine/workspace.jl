# The mask engine: one FFT of the input, then per subband a masked product and
# an inverse FFT (analysis); per group of subbands (`_MaskSet`) their sum, an
# FFT and a masked accumulation, then the post-factor and one inverse FFT
# (synthesis).
#
# Data are transformed on the `rfft` grid.  Complex data are split into their
# real and imaginary parts, two real channels that go through the same masks, so
# the transform is real-linear bit for bit and a complex input costs two real
# ones.  The spectra of the channels are stacked along a trailing axis.
#
# The subbands (groups) are independent, so both passes split them into
# contiguous chunks, one task per FFT buffer (`MultiscaleFilterBanks.foreach_chunk`);
# the synthesis gives each task after the first its own partial sum
# (`chunked_sum!`).  FFT plans are single-threaded.

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
        Td, Tf, N, S <: ShearletSystem, MS <: _MaskSet,
        C <: AbstractArray{Complex{Tf}}, CM <: AbstractMatrix{Complex{Tf}}, B <: AbstractArray{Complex{Tf}, N},
        V <: AbstractVector{Complex{Tf}}, R <: AbstractArray{Tf, N}, P, Q,
    }
    system::S
    dims::NTuple{N, Int}
    masks::MS
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
_nsubbands(ws::ShearletWorkspace) = length(ws.masks.ana)
# Whether a call threads: asked to, and not inside a caller's NestedThreading budget.
_threaded(threaded::Union{Bool, Nothing}) = loop_threaded(threaded !== false)

"""
    make_shearlet_workspace(T, dims, system; flags = FFTW.MEASURE) -> ShearletWorkspace

Build the masks, buffers and FFT plans of `system` for data of element type `T`
(real or complex) and size `dims`.  `flags` are the FFTW planner flags.
"""
function make_shearlet_workspace(::Type{T}, dims::NTuple{N, Int}, sys::ShearletSystem; flags::UInt32 = FFTW.MEASURE) where {T <: Number, N}
    return _make_workspace(T, dims, sys, flags, true)
end

# A host workspace, with buffers for one task per thread when `threaded`, else one.
function _make_workspace(::Type{T}, dims::NTuple{N, Int}, sys::ShearletSystem, flags::UInt32, threaded::Bool) where {T, N}
    Td = float(T)
    masks = _system_masks(sys, dims, real(Td))
    proto = Array{real(Td), N}(undef, ntuple(_ -> 0, N))
    nt = threaded ? n_tasks(proto, length(masks.ana)) : 1
    return _assemble_workspace(sys, dims, Td, masks, proto, nt) do rbuf, buf
        # FFTW's own threads off: the engine threads over subbands instead.  The
        # plans assume aligned arrays (`UNALIGNED` costs 1.3–1.8×); `_plannable`
        # routes any other array through an aligned buffer.
        with_fftw_threads(1) do
            plan_rfft(rbuf; flags), plan_brfft(buf, dims[1]; flags)
        end
    end
end

# A workspace around `masks`, with buffers like `proto` (an array of the filter
# type, host or device) for `nt` tasks and the FFT plans `plans(rbuf, buf)` makes
# for a real buffer and a spectrum buffer.
function _assemble_workspace(
        plans::F, sys::ShearletSystem, dims::NTuple{N, Int}, ::Type{Td}, masks::_MaskSet,
        proto::AbstractArray{Tf, N}, nt::Int
    ) where {F, N, Td, Tf}
    nch = _nchannels(Td)
    rd = _rfft_dims(dims)
    zeroed(T, sz) = fill!(similar(proto, T, sz), zero(T))
    X = zeroed(Complex{Tf}, (rd..., nch))
    # The inverse runs at most one task per synthesis group.
    partials = [similar(X) for _ in 2:min(nt, length(masks.groups))]
    bufs = [zeroed(Complex{Tf}, rd) for _ in 1:nt]
    rbufs = [[zeroed(Tf, dims) for _ in 1:nch] for _ in 1:nt]
    pfwd, pinv = plans(rbufs[1][1], bufs[1])
    Xm = reshape(X, :, nch)
    bufs_v = [vec(b) for b in bufs]
    return ShearletWorkspace{
        Td, Tf, N, typeof(sys), typeof(masks), typeof(X), typeof(Xm), eltype(bufs),
        eltype(bufs_v), eltype(eltype(rbufs)), typeof(pfwd), typeof(pinv),
    }(
        sys, dims, masks, X, Xm, partials, [reshape(p, :, nch) for p in partials],
        bufs, bufs_v, rbufs, pfwd, pinv, Tf(1 / prod(dims))
    )
end

# The masks of `system` at size `dims`, after `shearlet_layout` checked that the
# system supports that many dimensions.
function _system_masks(sys::ShearletSystem, dims::Tuple, ::Type{Tf}) where {Tf}
    n = length(shearlet_layout(sys, dims))
    masks = _shearlet_masks(sys, dims, Tf)
    @assert length(masks.ana) == n
    return masks
end

make_shearlet_workspace(::Type{T}, dims::Tuple, sys::ShearletSystem; kwargs...) where {T <: Number} =
    make_shearlet_workspace(T, Tuple(Int.(dims)), sys; kwargs...)

# The workspace for one call: `workspace` after checking it fits, else a new one.
function _workspace(workspace, sys::ShearletSystem, dims::Tuple, ::Type{Td}, threaded::Bool) where {Td}
    if workspace === nothing
        return _make_workspace(Td, Tuple(Int.(dims)), sys, FFTW.ESTIMATE, threaded)
    end
    workspace.dims == dims ||
        throw(DimensionMismatch("workspace was built for size $(workspace.dims), got $(dims)"))
    _nchannels(workspace) == _nchannels(Td) && real(eltype(workspace)) == real(Td) ||
        throw(ArgumentError("workspace was built for $(eltype(workspace)) data, got $(Td)"))
    workspace.system == sys || throw(ArgumentError("workspace was built for a different shearlet system"))
    return workspace
end

Base.eltype(::ShearletWorkspace{Td}) where {Td} = Td

# F = rfft of channel `c` of `x`, read by the plan `p` directly if it can, else
# through the real buffer `rb`.
function _rfft_channel!(F, p, x::AbstractArray, c::Int, rb::AbstractArray{Tf, N}) where {Tf, N}
    if _plannable(x, Tf, N)
        mul!(F, p, x)
    else
        @.. rb = _channel(x, c)
        mul!(F, p, rb)
    end
    return F
end

# Whether the FFT plans can read or write `a` in place: a dense, column-major
# array of the filter type (a whole array, or a slice of the stacked subbands),
# 16-byte aligned on the host like the buffers the plans were made for.
_plannable(a::AbstractArray, ::Type{Tf}, N::Int) where {Tf} =
    a isa StridedArray{Tf} && ndims(a) == N && strides(a) == Base.size_to_strides(1, size(a)...) && _fftw_aligned(a)

_fftw_aligned(a::AbstractArray) = parent(a) === a ? !(a isa Array) || FFTW.alignment_of(a) == 0 : _fftw_aligned_in(a, parent(a))
_fftw_aligned_in(a, ::Array) = FFTW.alignment_of(a) == 0
_fftw_aligned_in(a, p) = _fftw_aligned(p)   # device arrays: their FFT library needs no alignment

_channel(x::Real, ::Int) = x
_channel(x::Complex, c::Int) = c == 1 ? real(x) : imag(x)

# Analysis: subband i of every channel, handed to `emit(i, t)` once the task-t
# real buffers hold it (`ws.rbufs[t][c]`), unless `target(i, t, c)` names the
# array the inverse FFT writes to directly.
function _engine_forward!(target::F, emit::G, x::AbstractArray, ws::ShearletWorkspace{Td, Tf, N}, threaded::Bool) where {F, G, Td, Tf, N}
    nch = _nchannels(ws)
    for c in 1:nch
        Xc = selectdim(ws.X, N + 1, c)
        # Channel 2 is misaligned when the spectrum has an odd length in Float32.
        if _plannable(Xc, Complex{Tf}, N)
            _rfft_channel!(Xc, ws.pfwd, x, c, ws.rbufs[1][1])
        else
            copyto!(Xc, _rfft_channel!(ws.bufs[1], ws.pfwd, x, c, ws.rbufs[1][1]))
        end
    end
    s = ws.scale
    ana = ws.masks.ana
    foreach_chunk(length(ana), _ntasks(ws), threaded) do range, t
        Z, Zv = ws.bufs[t], ws.bufs_v[t]
        for i in range
            for c in 1:nch
                _mask_apply!(Z, Zv, ana[i], ws.X, ws.Xm, c, s)
                mul!(target(i, t, c), ws.pinv, Z)
            end
            emit(i, t)
        end
    end
    return nothing
end

# Synthesis into `x` from the subbands `source(i)` (arrays of the data size): per
# group, the subbands summed in space, one FFT and the group's mask; then the
# post-factor and one inverse FFT per channel.
function _engine_inverse!(source::S, x::AbstractArray, ws::ShearletWorkspace{Td, Tf, N}, threaded::Bool) where {S, Td, Tf, N}
    nch = _nchannels(ws)
    s = ws.scale
    ms = ws.masks
    ng = length(ms.groups)
    fill!(ws.X, zero(eltype(ws.X)))
    # Fewer groups than tasks: the idle tasks need no partial sums.
    partials = view(ws.partials, 1:(min(_ntasks(ws), ng) - 1))
    chunked_sum!(ws.X, partials, ng, threaded) do a, range, t
        am = t == 1 ? ws.Xm : ws.partials_m[t - 1]
        F, Fv = ws.bufs[t], ws.bufs_v[t]
        rbs = ws.rbufs[t]
        for g in range
            grp = ms.groups[g]
            if length(grp) == 1
                band = source(first(grp))
                for c in 1:nch
                    _rfft_channel!(F, ws.pfwd, band, c, rbs[c])
                    _mask_accumulate!(a, am, ms.syn[g], F, Fv, c, s)
                end
            else
                for i in grp
                    band = source(i)
                    for c in 1:nch
                        rb = rbs[c]
                        i == first(grp) ? (@.. rb = _channel(band, c)) : (@.. rb += _channel(band, c))
                    end
                end
                for c in 1:nch
                    mul!(F, ws.pfwd, rbs[c])
                    _mask_accumulate!(a, am, ms.syn[g], F, Fv, c, s)
                end
            end
        end
    end
    if ms.post !== nothing
        for c in 1:nch
            selectdim(ws.X, N + 1, c) .*= ms.post
        end
    end
    if nch == 1 && _plannable(x, Tf, N)
        mul!(x, ws.pinv, selectdim(ws.X, N + 1, 1))
    else
        for c in 1:nch
            Xc = selectdim(ws.X, N + 1, c)
            if _plannable(Xc, Complex{Tf}, N)
                mul!(ws.rbufs[1][c], ws.pinv, Xc)
            else
                mul!(ws.rbufs[1][c], ws.pinv, copyto!(ws.bufs[1], Xc))
            end
        end
        _combine!(x, ws.rbufs[1])
    end
    return x
end

# dst = the array of channels `r` (one real channel, or real and imaginary).
_combine!(dst::AbstractArray, r::AbstractVector) = length(r) == 1 ? (dst .= r[1]) : (dst .= complex.(r[1], r[2]))
