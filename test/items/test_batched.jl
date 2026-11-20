# The chunked paths (one batched FFT per chunk of subbands) that device
# workspaces take, exercised on the host: their code is generic over the array type.

@testmodule ChunkCases begin
    using Shearlets, FFTW, Random
    using MultiscaleFilterBanks: MaskBankEngine
    export cases, random_input, rechunk
    # One system per mask kind and synthesis shape.
    const cases = [
        (BandlimitedShearlets(scales = 2), (64, 48)),     # sparse masks, layered synthesis
        (NSSTShearlets(directions = [2, 3]), (64, 64)),   # dense masks, grouped synthesis
        (CompactShearlets(scales = 2), (64, 64)),         # dense masks, post-factor
        (BandlimitedShearlets(scales = 1), (16, 20, 24)),
        (CompactShearlets(scales = 1), (16, 16, 16)),     # factored masks
    ]
    random_input(rng, T, dims) =
        T <: Real ? randn(rng, Float32, dims...) : complex.(randn(rng, Float32, dims...), randn(rng, Float32, dims...))

    # `ws` with its engine rebuilt to run its analysis (and, with `synthesis`,
    # its synthesis) in chunks of at most `k` subbands, with the planners of a
    # device workspace, or FFTW's on the host.
    function rechunk(ws, k; synthesis = true)
        e = ws.engine
        N = length(ws.dims)
        planner, forward_planner, output = something(
            e.tuning,
            (
                (B, R) -> plan_brfft(B, ws.dims[1], 1:N; flags = FFTW.UNALIGNED),
                (R, B) -> plan_rfft(R, 1:N; flags = FFTW.UNALIGNED),
                zeros(eltype(ws), ws.dims),
            )
        )
        chunked = MaskBankEngine(
            e.banks, e.items, similar(e.X), e.pfwd, e.pinv; groups = e.pyramid_syn, post = e.post,
            pbwd = e.pbwd, scale = e.scale, split_complex = eltype(ws) <: Complex,
            planner, forward_planner = synthesis ? forward_planner : nothing, output,
            chunk_samples = k * length(e.X), tasks = 1
        )
        return Shearlets._with_engine(ws, chunked)
    end
end

@testitem "Chunked path matches the per-subband engine" tags = [:workspace] setup = [ChunkCases] begin
    using Random
    rng = Xoshiro(7)
    for (sys, dims) in ChunkCases.cases, T in (Float32, ComplexF32)
        x = ChunkCases.random_input(rng, T, dims)
        ws = make_workspace(sys, T, dims)
        ref = analyze(sys, x; workspace = ws)
        n = size(ref, ndims(ref))
        # One chunk of all, uneven chunks, and chunks of two.
        for k in (n, 5, 2)
            wb = ChunkCases.rechunk(ws, k)
            @test wb.engine.batch !== nothing
            @test maximum(c -> length(c.items), wb.engine.batch.chunks) <= k
            c = analyze(sys, x; workspace = wb)
            @test maximum(abs, c .- ref) <= 1.0f-5 * maximum(abs, ref)
            @test maximum(abs, synthesize(sys, c; workspace = wb) .- x) <= 1.0f-5 * maximum(abs, x)
        end
        # The streaming forms run one subband at a time on a chunked engine too.
        wb = ChunkCases.rechunk(ws, 5)
        cs = similar(ref)
        analyze((i, band) -> copyto!(selectdim(cs, ndims(cs), i), band), sys, x; workspace = wb)
        @test maximum(abs, cs .- ref) <= 1.0f-5 * maximum(abs, ref)
        y = synthesize!(i -> selectdim(ref, ndims(ref), i), similar(x), sys; workspace = wb)
        @test maximum(abs, y .- x) <= 1.0f-5 * maximum(abs, x)
    end
end

@testitem "Chunked path: adapted workspace and tuning" tags = [:workspace] setup = [ChunkCases] begin
    using Adapt, Random
    const S = Shearlets
    sys, dims = CompactShearlets(scales = 2), (64, 64)
    x = randn(Xoshiro(3), Float32, dims...)
    ws = make_workspace(sys, Float32, dims)
    ref = analyze(sys, x; workspace = ws)
    # Adapted to host arrays, a workspace threads over subbands, unchunked.
    hw = Adapt.adapt(Array, ws)
    @test hw.engine.batch === nothing && hw.engine.tuning === nothing
    @test analyze(sys, x; workspace = hw) ≈ ref
    # Tuning keeps a chunked path or drops it, and the transform is the same.
    dw = ChunkCases.rechunk(ws, 8)
    tw = S._tune(dw)
    @test analyze(sys, x; workspace = tw) ≈ ref
    @test synthesize(sys, analyze(sys, x; workspace = tw); workspace = tw) ≈ x
end
