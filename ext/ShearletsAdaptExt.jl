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
# parallel already.

using Shearlets: Shearlets, ShearletWorkspace, _MaskSet, _SparseMask, _DenseMask, _ProductMask
using Adapt: Adapt

# Arrays shared between masks (a factor common to many subbands, a synthesis mask
# that is an analysis mask) stay shared on the device.
_adapt_shared(to, a, cache::IdDict) = get!(() -> Adapt.adapt(to, a), cache, a)
_adapt_shared(to, ::Nothing, cache::IdDict) = nothing

_adapt_mask(to, m::_SparseMask, cache) = _SparseMask(_adapt_shared(to, m.idx, cache), _adapt_shared(to, m.val, cache))
_adapt_mask(to, m::_DenseMask, cache) = _DenseMask(_adapt_shared(to, m.val, cache))
_adapt_mask(to, m::_ProductMask, cache) = _ProductMask(_adapt_shared(to, m.f1, cache), _adapt_shared(to, m.f2, cache))

function Adapt.adapt_structure(to, ws::ShearletWorkspace{Td, Tf, N}) where {Td, Tf, N}
    cache = IdDict{Any, Any}()
    ms = ws.masks
    masks = _MaskSet(
        [_adapt_mask(to, m, cache) for m in ms.ana], [_adapt_mask(to, m, cache) for m in ms.syn],
        ms.groups, _adapt_shared(to, ms.post, cache),
    )
    proto = Adapt.adapt(to, Array{Tf, N}(undef, ntuple(_ -> 0, N)))
    nt = Shearlets.n_tasks(proto, length(ms.ana))
    return Shearlets._assemble_workspace(ws.system, ws.dims, Td, masks, proto, nt) do rbuf, buf
        Shearlets.plan_rfft(rbuf), Shearlets.plan_brfft(buf, ws.dims[1])
    end
end

end # module ShearletsAdaptExt
