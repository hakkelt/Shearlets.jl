module ShearletsAdaptExt

# Shearlet transforms on a device, in Adapt methods alone, plus the device
# routing of `make_workspace(sys, x)` (`Shearlets._to_device`).
#
# No kernel is needed and none is written here.  Every transform is an FFT, a
# masked product per subband and an inverse FFT, and every mask is applied by a
# broadcast (MultiscaleFilterBanks' mask kinds and `Shearlets._ProductMask`),
# all of which run on device arrays unchanged.  What the device needs is the
# masks moved across and the FFT plans rebuilt: FFTW plans cannot be moved, so
# `plan_rfft` / `plan_brfft` (AbstractFFTs generics) are called again on device
# buffers and dispatch to the backend's FFT library:
#
#     ws = make_workspace(system, xg)             # xg a device array, or
#     ws = Adapt.adapt(CuArray, make_workspace(system, Float32, dims))
#     c  = analyze(system, xg; workspace = ws)
#
# The device engine runs in one task (the device's FFTs are parallel already) and
# transforms the subbands in chunks, one batched FFT per chunk, where the FFT
# library has batched plans.

using Shearlets: Shearlets, ShearletWorkspace, _MaskSet, _ProductMask
using MultiscaleFilterBanks: SparseMask
using Adapt: Adapt
using FFTW: FFTW, plan_rfft, plan_brfft

# Arrays shared between masks (a factor common to many subbands, a synthesis mask
# that is an analysis mask) stay shared on the device.
_adapt_shared(to, a, cache::IdDict) = get!(() -> Adapt.adapt(to, a), cache, a)
_adapt_shared(to, ::Nothing, cache::IdDict) = nothing

_adapt_mask(to, m::SparseMask, cache) = SparseMask(m.dims, _adapt_shared(to, m.idx, cache), _adapt_shared(to, m.val, cache))
_adapt_mask(to, m::AbstractArray, cache) = _adapt_shared(to, m, cache)
_adapt_mask(to, m::_ProductMask, cache) = _ProductMask(_adapt_shared(to, m.f1, cache), _adapt_shared(to, m.f2, cache))

# Batched plans over the first N axes of a stack, or `nothing` where the FFT
# library has none.
function _batched(plan, args...)
    try
        return plan(args...)
    catch err
        err isa Union{MethodError, ArgumentError} || rethrow()
        @debug "Shearlets: no batched FFT plans for these arrays; transforming one subband at a time" exception = err
        return nothing
    end
end

function Adapt.adapt_structure(to, ws::ShearletWorkspace{Td, Tf, N}) where {Td, Tf, N}
    cache = IdDict{Any, Any}()
    ms = ws.masks
    ana = [_adapt_mask(to, m, cache) for m in ms.ana]
    # The synthesis masks stay the analysis masks.
    syn = ms.syn === ms.ana ? ana : [_adapt_mask(to, m, cache) for m in ms.syn]
    masks = _MaskSet(ana, syn, ms.groups, _adapt_shared(to, ms.post, cache))
    dims = ws.dims
    rb = Shearlets._zeros_like(Adapt.adapt(to, Array{Tf, N}(undef, ntuple(_ -> 0, N))), Tf, dims)
    # A host engine (adapted to `Array`) threads over subbands instead, with
    # plans for slices of any alignment where the host workspace has them.
    host = Shearlets._on_host(rb)
    plans = host ? Shearlets._host_plans(FFTW.ESTIMATE, ws.aligned) : (rb, X) -> (plan_rfft(rb), plan_brfft(X, dims[1]))
    planner(B, R) = _batched(plan_brfft, B, dims[1], 1:N)
    forward_planner(R, B) = _batched(plan_rfft, R, 1:N)
    engine = host ? Shearlets._engine(masks, Td, rb, plans) :
        Shearlets._engine(masks, Td, rb, plans; planner, forward_planner)
    buf() = Shearlets._zeros_like(rb, Td, dims)
    return ShearletWorkspace{Td}(ws.system, dims, ws.masks, engine, host && ws.aligned, buf(), buf())
end

# Adapts host arrays to arrays like `proto` (the storage of some device array).
struct _Like{A <: AbstractArray}
    proto::A
end

Adapt.adapt_storage(to::_Like, a::AbstractArray) = copyto!(similar(to.proto, eltype(a), size(a)), a)

_storage(x::AbstractArray) = (p = parent(x); p === x ? x : _storage(p))

Shearlets._to_device(ws::ShearletWorkspace, x::AbstractArray) = Adapt.adapt(_Like(_storage(x)), ws)

end # module ShearletsAdaptExt
