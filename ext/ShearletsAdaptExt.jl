module ShearletsAdaptExt

# Shearlet transforms on a device, in `Adapt.adapt_structure` methods alone.
#
# No kernel is needed and none is written here.  Every transform is an FFT, a
# masked product per subband and an inverse FFT, and every mask is applied by a
# broadcast: a gather/scatter over the support indices (`_SparseMask`), a dense
# product (`_DenseMask`) or a product of broadcastable factors (`_ProductMask`).
# All of these run on device arrays unchanged.  What the device needs is the
# masks moved across and the FFT plans rebuilt — FFTW plans cannot be moved, so
# `plan_rfft` / `plan_brfft` (AbstractFFTs generics) are called again on the
# adapted buffers and dispatch to the backend's FFT library:
#
#     ws = Adapt.adapt(CuArray, make_shearlet_workspace(Float32, dims, system))
#     c  = shearlet_forward(CuArray(x), system; workspace = ws)
#
# The device workspace runs its subbands in one task: the device's FFTs are
# parallel already.  It transforms them in chunks of several subbands, one batched
# FFT per chunk (`Shearlets._make_batch`), sized by `Shearlets._default_chunk`.

using Shearlets: Shearlets, ShearletWorkspace, _MaskSet, _SparseMask, _DenseMask, _ProductMask
using Adapt: Adapt

# Arrays shared between masks (a factor common to many subbands, a synthesis mask
# that is an analysis mask) stay shared on the device.
_adapt_shared(to, a, cache::IdDict) = get!(() -> Adapt.adapt(to, a), cache, a)
_adapt_shared(to, ::Nothing, cache::IdDict) = nothing

# The dense masks become the columns of one device matrix, so a chunk of them is
# a view of it (`Shearlets._dense_stack`).
function _stack_dense!(to, ms::_MaskSet, cache::IdDict)
    vals = unique(objectid, [m.val for m in vcat(ms.ana, ms.syn) if m isa _DenseMask])
    isempty(vals) && return cache
    M = Adapt.adapt(to, reduce(hcat, vals))
    for (j, v) in enumerate(vals)
        cache[v] = view(M, :, j)
    end
    return cache
end

_adapt_mask(to, m::_SparseMask, cache) = _SparseMask(_adapt_shared(to, m.idx, cache), _adapt_shared(to, m.val, cache))
_adapt_mask(to, m::_DenseMask, cache) = _DenseMask(_adapt_shared(to, m.val, cache))
_adapt_mask(to, m::_ProductMask, cache) = _ProductMask(_adapt_shared(to, m.f1, cache), _adapt_shared(to, m.f2, cache))

function Adapt.adapt_structure(to, ws::ShearletWorkspace{Td, Tf, N}) where {Td, Tf, N}
    cache = IdDict{Any, Any}()
    ms = ws.masks
    _stack_dense!(to, ms, cache)
    masks = _MaskSet(
        [_adapt_mask(to, m, cache) for m in ms.ana], [_adapt_mask(to, m, cache) for m in ms.syn],
        ms.groups, _adapt_shared(to, ms.post, cache),
    )
    proto = Adapt.adapt(to, Array{Tf, N}(undef, ntuple(_ -> 0, N)))
    nt = Shearlets.n_tasks(proto, length(ms.ana))
    dws = Shearlets._assemble_workspace(ws.system, ws.dims, Td, masks, proto, nt) do rbuf, buf
        Shearlets.plan_rfft(rbuf), Shearlets.plan_brfft(buf, ws.dims[1])
    end
    k = Shearlets._default_chunk(dws)
    k < 2 && return dws
    # `nothing`, the per-subband engine, when the FFT library has no batched plans.
    return Shearlets._with_batch(dws, Shearlets._make_batch(dws, k))
end

end # module ShearletsAdaptExt
