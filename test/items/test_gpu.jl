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
    isempty(gpu_backends(; include_jlarrays = false)) &&
        GPUEnv.activate(persist = true, include_jlarrays = false)
    const fft_backends = gpu_backends(; include_jlarrays = false, supports_fftw = true)
end

@testitem "GPU Adapt extension is loaded" tags = [:gpu] setup = [GPUBackends] begin
    using Adapt
    @test Base.get_extension(Shearlets, :ShearletsAdaptExt) !== nothing
end

@testitem "GPU shearlet transforms match the CPU" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    rng = Xoshiro(42)
    cases = [
        (BandlimitedShearlets(scales = 2), (64, 48)),
        (NSSTShearlets(directions = [2, 3]), (64, 64)),
        (CompactShearlets(scales = 2), (64, 64)),
        (BandlimitedShearlets(scales = 1), (16, 20, 24)),
        (CompactShearlets(scales = 1), (16, 16, 16)),
    ]
    for (sys, dims) in cases, T in (Float32, ComplexF32)
        x = T <: Real ? randn(rng, Float32, dims...) : complex.(randn(rng, Float32, dims...), randn(rng, Float32, dims...))
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
        end
    end
end
