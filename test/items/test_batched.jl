# The chunked path (one batched FFT per chunk of subbands) that device workspaces
# take, exercised on the host: its code is generic over the array type.

@testmodule ChunkCases begin
    using Shearlets, Random
    export cases, random_input
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
end

@testitem "Chunked path matches the per-subband engine" tags = [:workspace] setup = [ChunkCases] begin
    using Random
    const S = Shearlets
    rng = Xoshiro(7)
    for (sys, dims) in ChunkCases.cases, T in (Float32, ComplexF32)
        x = ChunkCases.random_input(rng, T, dims)
        ws = make_shearlet_workspace(T, dims, sys)
        ref = shearlet_forward(x, sys; workspace = ws)
        n = size(ref, ndims(ref))
        # One chunk of all, uneven chunks, and chunks of one.
        for k in (n, 5, 1)
            wb = S._with_batch(ws, S._make_batch(ws, k))
            @test maximum(length, wb.batch.chunks) <= k
            c = shearlet_forward(x, sys; workspace = wb)
            @test maximum(abs, c .- ref) <= 1.0f-5 * maximum(abs, ref)
            @test maximum(abs, shearlet_inverse(c, sys; workspace = wb) .- x) <= 1.0f-5 * maximum(abs, x)
            @test (wb.batch.syn === nothing) == (sys isa NSSTShearlets)
        end
        # The streaming forms take the chunked path too.
        wb = S._with_batch(ws, S._make_batch(ws, 5))
        cs = similar(ref)
        shearlet_forward((i, band) -> copyto!(selectdim(cs, ndims(cs), i), band), x, sys; workspace = wb)
        @test maximum(abs, cs .- ref) <= 1.0f-5 * maximum(abs, ref)
        y = shearlet_inverse(i -> selectdim(ref, ndims(ref), i), sys, dims; T = T, workspace = wb)
        @test maximum(abs, y .- x) <= 1.0f-5 * maximum(abs, x)
    end
end

@testitem "Chunked path: device-style workspace and tuning" tags = [:workspace] begin
    using Adapt, Random
    const S = Shearlets
    sys, dims = CompactShearlets(scales = 2), (64, 64)
    x = randn(Xoshiro(3), Float32, dims...)
    ws = make_shearlet_workspace(Float32, dims, sys)
    ref = shearlet_forward(x, sys; workspace = ws)
    # The extension stacks the dense masks into one matrix, and the chunks are
    # views of it; it chunks device workspaces only.
    hw = Adapt.adapt(Array, ws)
    @test hw.batch === nothing && hw.masks.syn === hw.masks.ana
    @test hw.masks.ana[1].val isa SubArray
    dw = S._with_batch(hw, S._make_batch(hw, S._default_chunk(hw)))
    @test parent(dw.batch.ana[1]) === parent(dw.masks.ana[1].val)
    @test dw.batch.syn === dw.batch.ana
    @test shearlet_forward(x, sys; workspace = dw) ≈ ref
    # Tuning keeps a chunked path or drops it, and the transform is the same.
    tw = S._tune_batch(dw; candidates = [1, 2, 8])
    sizes = [maximum(length, S.even_chunks(S._nsubbands(dw), k)) for k in (1, 2, 8)]
    @test S._chunk_size(tw.batch) in (sizes..., S._chunk_size(dw.batch))
    @test shearlet_forward(x, sys; workspace = tw) ≈ ref
end
