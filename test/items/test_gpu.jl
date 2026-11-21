# GPU tests for ShearletsAdaptExt, on every backend GPUEnv finds with an
# FFTW-compatible FFT library (CUDA, AMDGPU).  The device path is broadcasts and
# FFTs only, so these items need nothing but `Adapt`; on a host with no such
# backend they are vacuously true.

@testmodule GPUBackends begin
    using GPUEnv
    using Adapt
    export fft_backends, Adapt
    # A test item may run outside `runtests.jl` (an IDE, a language server);
    # activate the overlay here too, which only loads it when it exists.
    isempty(gpu_backends(; include_cpu_backends = false)) &&
        GPUEnv.activate(persist = true, include_cpu_backends = false)
    const fft_backends = gpu_backends(; include_cpu_backends = false, supports_fftw = true)
end

@testitem "GPU Adapt extension is loaded" tags = [:gpu] setup = [GPUBackends] begin
    using Adapt
    @test Base.get_extension(Shearlets, :ShearletsAdaptExt) !== nothing
end

@testitem "GPU shearlet transforms match the CPU" tags = [:gpu] setup = [GPUBackends, ChunkCases] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    rng = Xoshiro(42)
    for (sys, dims) in ChunkCases.cases, T in (Float32, ComplexF32)
        x = ChunkCases.random_input(rng, T, dims)
        cpu = make_workspace(sys, T, dims)
        ref = analyze(sys, x; workspace = cpu)
        for backend in GPUBackends.fft_backends
            xg = to_gpu(backend, x)
            ws = Adapt.adapt(typeof(xg).name.wrapper, cpu)
            c = analyze(sys, xg; workspace = ws)
            # A workspace made from device data lives on its device.
            wd = make_workspace(sys, xg)
            @test wd.xbuf isa typeof(xg).name.wrapper
            @test maximum(abs, Array(analyze(wd, xg)) .- ref) < 1.0f-4 * maximum(abs, ref)
            # `flags` and `tasks` are ignored on a device; `prewarm` runs there.
            wp = make_workspace(sys, xg; flags = Shearlets.FFTW.ESTIMATE, tasks = 1, tune = false, prewarm = true)
            @test wp.xbuf isa typeof(xg).name.wrapper
            @test maximum(abs, Array(analyze(wp, xg)) .- ref) < 1.0f-4 * maximum(abs, ref)
            @test maximum(abs, Array(c) .- ref) < 1.0f-4 * maximum(abs, ref)
            @test maximum(abs, Array(synthesize(sys, c; workspace = ws)) .- x) < 1.0f-4 * maximum(abs, x)
            @test_throws ArgumentError analyze(sys, xg)
            # The streaming inverse returns device data.
            y = synthesize!(i -> selectdim(c, ndims(c), i), similar(xg), sys; workspace = ws)
            @test y isa typeof(xg).name.wrapper
            @test maximum(abs, Array(y) .- x) < 1.0f-4 * maximum(abs, x)
            # CUDA has batched plans, so the workspace is chunked (one chunk at these sizes).
            typeof(xg).name.name === :CuArray && @test ws.engine.batch !== nothing
        end
    end
end

@testitem "GPU chunked transforms match the CPU" tags = [:gpu] setup = [GPUBackends, ChunkCases] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    rng = Xoshiro(5)
    for (sys, dims) in ChunkCases.cases, T in (Float32, ComplexF32)
        x = ChunkCases.random_input(rng, T, dims)
        cpu = make_workspace(sys, T, dims)
        ref = analyze(sys, x; workspace = cpu)
        for backend in GPUBackends.fft_backends
            xg = to_gpu(backend, x)
            ws = Adapt.adapt(typeof(xg).name.wrapper, cpu)
            # Uneven chunks with a short last one, and chunks of two.
            for k in (5, 2)
                wb = ChunkCases.rechunk(ws, k)
                c = analyze(sys, xg; workspace = wb)
                @test maximum(abs, Array(c) .- ref) < 1.0f-4 * maximum(abs, ref)
                @test maximum(abs, Array(synthesize(sys, c; workspace = wb)) .- x) < 1.0f-4 * maximum(abs, x)
            end
            tw = make_workspace(sys, xg; tune = true)
            @test tw.engine.batch === nothing || tw.engine.batch.B[1] isa typeof(xg).name.wrapper
            c = analyze(sys, xg; workspace = tw)
            @test maximum(abs, Array(c) .- ref) < 1.0f-4 * maximum(abs, ref)
            @test maximum(abs, Array(synthesize(sys, c; workspace = tw)) .- x) < 1.0f-4 * maximum(abs, x)
        end
    end
end

@testitem "GPU NSST at 512²" tags = [:gpu] setup = [GPUBackends, ChunkCases] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    sys, dims = NSSTShearlets(direction_levels = [2, 3]), (512, 512)
    rng = Xoshiro(9)
    for T in (Float32, ComplexF32)
        x = ChunkCases.random_input(rng, T, dims)
        cpu = make_workspace(sys, T, dims)
        ref = analyze(sys, x; workspace = cpu)
        for backend in GPUBackends.fft_backends
            ws = Adapt.adapt(typeof(to_gpu(backend, x)).name.wrapper, cpu)
            c = to_gpu(backend, ref)
            @test maximum(abs, Array(synthesize(sys, c; workspace = ws)) .- x) < 1.0f-4 * maximum(abs, x)
            # The synthesis one subband at a time sums each level's stored
            # slices on the device in one pass.
            wl = ChunkCases.rechunk(ws, 8; synthesis = false)
            @test wl.engine.batch === nothing || wl.engine.batch.smasks === nothing
            @test maximum(abs, Array(synthesize(sys, c; workspace = wl)) .- x) < 1.0f-4 * maximum(abs, x)
        end
    end
end
