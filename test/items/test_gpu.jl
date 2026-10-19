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
        cpu = make_shearlet_workspace(T, dims, sys)
        ref = shearlet_forward(x, sys; workspace = cpu)
        for backend in GPUBackends.fft_backends
            xg = to_gpu(backend, x)
            ws = Adapt.adapt(typeof(xg).name.wrapper, cpu)
            c = shearlet_forward(xg, sys; workspace = ws)
            @test maximum(abs, Array(c) .- ref) < 1.0f-4 * maximum(abs, ref)
            @test maximum(abs, Array(shearlet_inverse(c, sys; workspace = ws)) .- x) < 1.0f-4 * maximum(abs, x)
            @test_throws ArgumentError shearlet_forward(xg, sys)
            # The streaming inverse returns device data.
            y = shearlet_inverse(i -> selectdim(c, ndims(c), i), sys, dims; T = T, workspace = ws)
            @test y isa typeof(xg).name.wrapper
            @test maximum(abs, Array(y) .- x) < 1.0f-4 * maximum(abs, x)
            # CUDA has batched plans, so the workspace is chunked (one chunk at these sizes).
            typeof(xg).name.name === :CuArray && @test ws.batch !== nothing
        end
    end
end

@testitem "GPU chunked transforms match the CPU" tags = [:gpu] setup = [GPUBackends, ChunkCases] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    const S = Shearlets
    rng = Xoshiro(5)
    for (sys, dims) in ChunkCases.cases, T in (Float32, ComplexF32)
        x = ChunkCases.random_input(rng, T, dims)
        cpu = make_shearlet_workspace(T, dims, sys)
        ref = shearlet_forward(x, sys; workspace = cpu)
        for backend in GPUBackends.fft_backends
            xg = to_gpu(backend, x)
            ws = Adapt.adapt(typeof(xg).name.wrapper, cpu)
            # Uneven chunks with a short last one, and chunks of two.
            for k in (5, 2)
                wb = S._with_batch(ws, S._make_batch(ws, k))
                c = shearlet_forward(xg, sys; workspace = wb)
                @test maximum(abs, Array(c) .- ref) < 1.0f-4 * maximum(abs, ref)
                @test maximum(abs, Array(shearlet_inverse(c, sys; workspace = wb)) .- x) < 1.0f-4 * maximum(abs, x)
            end
            tw = tune_shearlet_workspace(ws)
            c = shearlet_forward(xg, sys; workspace = tw)
            @test maximum(abs, Array(c) .- ref) < 1.0f-4 * maximum(abs, ref)
            @test maximum(abs, Array(shearlet_inverse(c, sys; workspace = tw)) .- x) < 1.0f-4 * maximum(abs, x)
        end
    end
end

@testitem "GPU NSST group sums by reduction" tags = [:gpu] setup = [GPUBackends, ChunkCases] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    # Large enough that the stacked inverse sums each level by one reduction.
    sys, dims = NSSTShearlets(directions = [2, 3]), (512, 512)
    @test prod(dims) >= Shearlets._GROUP_SUM_SAMPLES
    rng = Xoshiro(9)
    for T in (Float32, ComplexF32)
        x = ChunkCases.random_input(rng, T, dims)
        cpu = make_shearlet_workspace(T, dims, sys)
        ref = shearlet_forward(x, sys; workspace = cpu)
        for backend in GPUBackends.fft_backends
            ws = Adapt.adapt(typeof(to_gpu(backend, x)).name.wrapper, cpu)
            c = to_gpu(backend, ref)
            @test maximum(abs, Array(shearlet_inverse(c, sys; workspace = ws)) .- x) < 1.0f-4 * maximum(abs, x)
        end
    end
end
