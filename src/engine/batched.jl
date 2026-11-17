# Chunked transforms, for devices.
#
# On a device the per-subband engine is bound by launches: a subband is a few
# small kernels and one FFT too small to fill the device.  The chunked path takes
# k consecutive subbands at a time, stacks their masked spectra, applies their
# masks in one fused kernel where the mask kind allows and runs one batched FFT
# over the stack:
#
#     analysis:   B[:, j] = s · A_{r_j} · X_c,   coeffs[.., r]_c = brfft(B)
#     synthesis:  B = rfft(coeffs[.., r]_c),   X_c += Σ_j s · S_{r_j} · B[:, j]
#
# The synthesis takes this path only when every group is one subband (not the
# NSST, whose levels already cost one FFT each).  The FFTs of real data read and
# write the coefficients in place where the plans can; complex data, and slices
# the plans cannot take, go through the real stacks R.  Chunks have k or k - 1
# subbands, with a pair of plans each, over the leading columns of the stacks.

# The masks of a chunk, fused:
# - `_SparseChunk`: the supports of its subbands, `pts` on the grid and `src` in
#   the stack (offset to the subband's column), one gather/scatter.  The entries
#   are sorted into layers of distinct grid points, so the synthesis scatter-adds
#   a layer per broadcast without collisions; overlapping supports make only a
#   few layers.
# - a matrix: dense masks as its columns, one broadcast;
# - `_MaskRange`: other masks, one product per subband.
struct _SparseChunk{I <: AbstractVector{Int32}, V <: AbstractVector}
    pts::I                          # grid indices
    src::I                          # the same in the stack
    val::V
    layers::Vector{UnitRange{Int}}  # entries of distinct grid points
end

struct _MaskRange{M <: AbstractVector}
    masks::M
end

"""
    _Batch

The chunked path of a workspace: the chunks of subbands, the stacks and the
batched plans over their first N axes, and the fused masks of every chunk
(`syn === nothing` when the synthesis is not chunked).
"""
struct _Batch{CB <: AbstractArray, CM <: AbstractMatrix, V <: AbstractVector, RB <: AbstractArray, P, Q, A, S}
    chunks::Vector{UnitRange{Int}}
    B::CB               # (rfft dims..., k): masked spectra, or the spectra of the subbands (synthesis)
    Bm::CM              # the same memory as (points, k)
    ones::V             # k ones: the column sum of B is a matrix-vector product
    R::Vector{RB}       # per channel, (dims..., k): the chunk's subbands in space
    pinv::Vector{P}     # batched brfft, B → R, for chunks of k and k - 1 (`_plan`)
    pfwd::Q             # the same for rfft, R → B (`nothing` without a chunked synthesis)
    ana::Vector{A}
    syn::S
end

_chunk_size(bt::_Batch) = size(bt.B, ndims(bt.B))
_chunk_size(::Nothing) = 1

# The plan of `plans` for a chunk of `l` subbands.
_plan(bt::_Batch, plans::Vector, l::Int) = plans[_chunk_size(bt) - l + 1]

"""
    _make_batch(ws, k) -> Union{_Batch, Nothing}

The chunked path of the workspace `ws` with chunks of at most `k` subbands; its
buffers are like `ws`'s (host or device) and its plans are AbstractFFTs plans of
them.  `nothing` when the FFT library has no batched plans.
"""
function _make_batch(ws::ShearletWorkspace{Td, Tf, N}, k::Int) where {Td, Tf, N}
    chunks = even_chunks(_nsubbands(ws), k)
    k = maximum(length, chunks)
    rd = _rfft_dims(ws.dims)
    L = prod(rd)
    L * k < typemax(Int32) || throw(ArgumentError("a chunk of $(k) subbands has too many samples for Int32 indices"))
    ms = ws.masks
    chunked_syn = all(g -> length(g) == 1, ms.groups)
    proto = ws.rbufs[1][1]
    B = _zeros_like(proto, Complex{Tf}, (rd..., k))
    R = [_zeros_like(proto, Tf, (ws.dims..., k)) for _ in 1:_nchannels(ws)]
    lens = k:-1:minimum(length, chunks)
    plans = try
        with_fftw_threads(1) do
            [plan_brfft(leading_slices(B, l), ws.dims[1], 1:N) for l in lens],
                chunked_syn ? [plan_rfft(leading_slices(R[1], l), 1:N) for l in lens] : nothing
        end
    catch err
        err isa Union{MethodError, ArgumentError} || rethrow()
        @debug "Shearlets: no batched FFT plans for these arrays; transforming one subband at a time" exception = err
        foreach(finalize, (B, R...))
        return nothing
    end
    ana = _chunk_masks(ms.ana, chunks, L)
    syn = !chunked_syn ? nothing : ms.syn === ms.ana ? ana : _chunk_masks(ms.syn, chunks, L)
    ones = fill!(similar(B, k), one(eltype(B)))
    return _Batch(chunks, B, reshape(B, L, k), ones, R, plans..., ana, syn)
end

function _chunk_masks(masks::Vector{<:_SparseMask}, chunks, L::Int)
    count = zeros(Int32, L)
    return map(chunks) do r
        sub = masks[r]
        pts = reduce(vcat, [Array(m.idx) for m in sub])
        src = reduce(vcat, [Array(m.idx) .+ Int32((j - 1) * L) for (j, m) in enumerate(sub)])
        val = reduce(vcat, [Array(m.val) for m in sub])
        # An entry's layer: how many entries before it share its grid point.
        layer = [count[p] += Int32(1) for p in pts]
        count[pts] .= 0
        perm = sortperm(layer)
        sorted = layer[perm]
        layers = [searchsortedfirst(sorted, l):searchsortedlast(sorted, l) for l in 1:maximum(sorted; init = 0)]
        idx = first(sub).idx
        _SparseChunk(_like(idx, pts[perm]), _like(idx, src[perm]), _like(first(sub).val, val[perm]), layers)
    end
end

function _chunk_masks(masks::Vector{<:_DenseMask}, chunks, L::Int)
    M = _dense_stack(masks)
    return [view(M, :, r) for r in chunks]
end

_chunk_masks(masks::Vector, chunks, L::Int) = [_MaskRange(view(masks, r)) for r in chunks]

# The host vector `v` in an array like `a` (host or device).
_like(a::AbstractArray, v::Vector) = copyto!(similar(a, eltype(v), length(v)), v)

# The dense masks as the columns of one matrix: the matrix they are views of
# when they are its consecutive columns (adapted masks, see the Adapt
# extension), else a new one.
function _dense_stack(masks::Vector{<:_DenseMask})
    v1 = first(masks).val
    if v1 isa SubArray && ndims(parent(v1)) == 2
        M = parent(v1)
        j1 = parentindices(v1)[2]
        stacked = all(enumerate(masks)) do (j, m)
            v = m.val
            v isa SubArray && parent(v) === M && parentindices(v) == (Base.Slice(axes(M, 1)), j1 + j - 1)
        end
        stacked && return view(M, :, j1:(j1 + length(masks) - 1))
    end
    return reduce(hcat, [m.val for m in masks])
end

# The leading columns of the stack = s · mask · channel c of the spectra.
function _chunk_analysis!(bt::_Batch, ch::_SparseChunk, ws, c::Int, s)
    fill!(bt.Bm, zero(eltype(bt.Bm)))
    # Unchecked, as in `_mask_apply!`: the indices lie in the grid and the stack by construction.
    @inbounds view(bt.Bm, ch.src) .= (ch.val .* s) .* view(view(ws.Xm, :, c), ch.pts)
    return bt.B
end

function _chunk_analysis!(bt::_Batch, M::AbstractMatrix, ws, c::Int, s)
    view(bt.Bm, :, 1:size(M, 2)) .= (s .* M) .* view(ws.Xm, :, c)
    return bt.B
end

function _chunk_analysis!(bt::_Batch, ch::_MaskRange, ws, c::Int, s)
    for (j, m) in enumerate(ch.masks)
        _mask_apply!(selectdim(bt.B, ndims(bt.B), j), view(bt.Bm, :, j), m, ws.X, ws.Xm, c, s)
    end
    return bt.B
end

# Channel c of the synthesis sum += Σ_j s · S_{r_j} · B[:, j], with the spectra
# of the chunk's subbands in the leading columns of B.
function _chunk_accumulate!(ws, bt::_Batch, ch::_SparseChunk, c::Int, s)
    Xc = view(ws.Xm, :, c)
    for l in ch.layers
        pts, src, val = view(ch.pts, l), view(ch.src, l), view(ch.val, l)
        @inbounds view(Xc, pts) .+= (val .* s) .* view(bt.Bm, src)
    end
    return ws.X
end

function _chunk_accumulate!(ws, bt::_Batch, M::AbstractMatrix, c::Int, s)
    n = size(M, 2)
    Bn = view(bt.Bm, :, 1:n)
    Bn .*= s .* M
    # The column sum as a matrix-vector product, added in place (2× faster than
    # `sum!` and an addition on an A100 at 256²).
    mul!(view(ws.Xm, :, c), Bn, view(bt.ones, 1:n), true, true)
    return ws.X
end

function _chunk_accumulate!(ws, bt::_Batch, ch::_MaskRange, c::Int, s)
    for (j, m) in enumerate(ch.masks)
        _mask_accumulate!(ws.X, ws.Xm, m, selectdim(bt.B, ndims(bt.B), j), view(bt.Bm, :, j), c, s)
    end
    return ws.X
end

# Analysis in chunks: the batched inverse FFT of channel c of the subbands r
# writes `target(r, c)`, then `emit(r)` runs.
function _batched_forward!(target::T, emit::E, x::AbstractArray, ws::ShearletWorkspace) where {T, E}
    bt = ws.batch
    _spectra!(ws, x)
    for (q, r) in enumerate(bt.chunks)
        l = length(r)
        for c in 1:_nchannels(ws)
            _chunk_analysis!(bt, bt.ana[q], ws, c, ws.scale)
            mul!(target(r, c), _plan(bt, bt.pinv, l), leading_slices(bt.B, l))
        end
        emit(r)
    end
    return nothing
end

# Synthesis in chunks into `x`; `stage(r)` returns the subbands r of each
# channel, as real arrays of size (dims..., length(r)).
function _batched_inverse!(stage::S, x::AbstractArray, ws::ShearletWorkspace) where {S}
    bt = ws.batch
    fill!(ws.X, zero(eltype(ws.X)))
    for (q, r) in enumerate(bt.chunks)
        l = length(r)
        sources = stage(r)
        for c in 1:_nchannels(ws)
            mul!(leading_slices(bt.B, l), _plan(bt, bt.pfwd, l), sources[c])
            _chunk_accumulate!(ws, bt, bt.syn[q], c, ws.scale)
        end
    end
    return _finish_inverse!(x, ws)
end

_chunked_synthesis(ws::ShearletWorkspace) = ws.batch !== nothing && ws.batch.syn !== nothing

# The columns `js` (a range, or one) of every channel's stack `bt.R`.
_columns(bt::_Batch, js) = [selectdim(R, ndims(R), js) for R in bt.R]

# Whether the batched FFTs read and write the subbands r of `coeffs` in place.
_in_place(coeffs::AbstractArray, r, ws::ShearletWorkspace{Td, Tf, N}) where {Td, Tf, N} =
    Td <: Real && _plannable(_band(coeffs, r), Tf, N + 1)

function _batched_forward_into!(coeffs::AbstractArray, x::AbstractArray, ws::ShearletWorkspace)
    bt = ws.batch
    target(r, c) = _in_place(coeffs, r, ws) ? _band(coeffs, r) : leading_slices(bt.R[c], length(r))
    emit(r) = (_in_place(coeffs, r, ws) || _combine!(_band(coeffs, r), _columns(bt, 1:length(r))); nothing)
    _batched_forward!(target, emit, x, ws)
    return coeffs
end

function _batched_inverse_from!(x::AbstractArray, coeffs::AbstractArray, ws::ShearletWorkspace)
    bt = ws.batch
    function stage(r)
        bands = _band(coeffs, r)
        _in_place(coeffs, r, ws) && return (bands,)
        Rs = _columns(bt, 1:length(r))
        for (c, Rc) in enumerate(Rs)
            # A copy is faster on a device than the broadcast of `_channel`.
            eltype(bands) <: Real ? copyto!(Rc, bands) : (@.. Rc = _channel(bands, c))
        end
        return Rs
    end
    return _batched_inverse!(stage, x, ws)
end

# Choosing the chunk size.  Without measuring, a chunk holds at most
# `DEFAULT_CHUNK_SAMPLES` spectrum samples (MultiscaleFilterBanks' default for batched
# device transforms, chosen on these subbands).  Measuring times every chunk size
# up to `_BATCH_MAX_SAMPLES` against the per-subband engine.

const _BATCH_MAX_SAMPLES = 2^23

# The chunk size used without measuring; below 2 the per-subband engine runs.
_default_chunk(ws::ShearletWorkspace) = clamp(DEFAULT_CHUNK_SAMPLES ÷ prod(_rfft_dims(ws.dims)), 1, _nsubbands(ws))

# The chunk sizes worth measuring: 1 (the per-subband engine), powers of two,
# the default and the largest that fits.
function _chunk_candidates(ws::ShearletWorkspace)
    kmax = min(_nsubbands(ws), _BATCH_MAX_SAMPLES ÷ prod(_rfft_dims(ws.dims)))
    return sort!(unique!([1; filter(<(kmax), 2 .^ (1:30)); _default_chunk(ws); kmax]))
end

"""
    tune_shearlet_workspace(workspace) -> ShearletWorkspace

The device `workspace` (see [`make_shearlet_workspace`](@ref)) with the chunk
size measured fastest on this device.  A device workspace transforms its
subbands in chunks, one batched FFT per chunk; by default the chunk size follows
from the grid size alone.  This times the workspace as it is, one subband at a
time, and every chunk size worth trying, on a forward and an inverse transform
each, and keeps the current setting unless another is 5% faster.  It takes a
fraction of a second once compiled, and holds a coefficient array and the
stacks of one more chunk size on the device meanwhile.  The transform itself is
unchanged.

```julia
ws = tune_shearlet_workspace(Adapt.adapt(CuArray, make_shearlet_workspace(Float32, dims, system)))
```

A host workspace is returned unchanged: it threads over subbands instead.
"""
tune_shearlet_workspace(ws::ShearletWorkspace) = _on_host(ws.rbufs[1][1]) ? ws : _tune_batch(ws)

"""
    _tune_batch(ws; candidates) -> ShearletWorkspace

`ws` with the fastest of its own chunked path and those of the `candidates`
chunk sizes (1: the per-subband engine), each timed on a forward and an inverse
transform of zeros through a coefficient array.  A candidate replaces the
fastest so far only when 5% faster, so noise keeps the current path.
"""
function _tune_batch(ws::ShearletWorkspace{Td}; candidates = _chunk_candidates(ws)) where {Td}
    x = _zeros_like(ws.rbufs[1][1], Td, ws.dims)
    c = _zeros_like(ws.rbufs[1][1], Td, (ws.dims..., _nsubbands(ws)))
    best, tbest = ws.batch, _time_transforms(ws, x, c)
    # Candidates that give the same chunks are timed once.
    for k in unique!([maximum(length, even_chunks(_nsubbands(ws), k)) for k in candidates])
        k == _chunk_size(ws.batch) && continue
        bt = k < 2 ? nothing : _make_batch(ws, k)
        t = _time_transforms(_with_batch(ws, bt), x, c)
        if t < 0.95 * tbest
            best === ws.batch || _release!(best)
            best, tbest = bt, t
        else
            _release!(bt)
        end
    end
    finalize(x)
    finalize(c)
    return _with_batch(ws, best)
end

# The shortest of a few forward-and-inverse rounds, in seconds, after a warm-up.
function _time_transforms(ws::ShearletWorkspace, x::AbstractArray, c::AbstractArray)
    round!() = (_forward_into!(c, x, ws, false); _inverse_from!(x, c, ws, false); _sync(x))
    round!()
    return minimum(_ -> (t = time_ns(); round!(); (time_ns() - t) / 1.0e9), 1:5)
end

# Waits for the device: reads one element of `a` back to the host.
_sync(a::AbstractArray) = (copyto!(Vector{eltype(a)}(undef, 1), 1, vec(a), 1, 1); nothing)

# Frees the buffers of a chunked path that is not kept (device arrays free on
# finalisation, more than once harmlessly; the dense chunks are views of the
# workspace's masks).
function _release!(bt::_Batch)
    foreach(finalize, (bt.B, bt.ones, bt.R..., bt.pinv...))
    bt.pfwd === nothing || foreach(finalize, bt.pfwd)
    for chunks in (bt.ana, something(bt.syn, ())), ch in chunks
        ch isa _SparseChunk && foreach(finalize, (ch.pts, ch.src, ch.val))
    end
    return nothing
end

_release!(::Nothing) = nothing
