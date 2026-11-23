@testitem "MultiscaleTransformsBase conformance" tags = [:interface] begin
    using MultiscaleTransformsBase.Testing: test_transform
    using Random
    rng = Xoshiro(11)
    for (sys, dims, other) in (
            (BandlimitedShearlets(scales = 2), (32, 40), (32, 32)),
            (NSSTShearlets(direction_levels = [2, 3]), (32, 40), (32, 32)),
            (CompactShearlets(scales = 2), (33, 32), (32, 32)),
            (BandlimitedShearlets(scales = 1), (12, 16, 10), (8, 8, 8)),
            (CompactShearlets(scales = 1), (12, 16, 10), (8, 8, 8)),
        )
        test_transform(sys, randn(rng, dims...); rtol = 1.0e-10, wrong_size = other)
        test_transform(sys, randn(rng, ComplexF64, dims...); rtol = 1.0e-10, exact_real_linear = true, wrong_size = other)
        test_transform(sys, randn(rng, Float32, dims...); wrong_size = other)
        test_transform(sys, randn(rng, ComplexF32, dims...); exact_real_linear = true)
    end
end

@testitem "Interface: accessors, workspace and ambiguities" tags = [:interface, :workspace] begin
    using MultiscaleTransformsBase, Random, Test
    sys = CompactShearlets(scales = 2)
    @test sys isa AbstractMultiscaleTransform
    x = randn(Xoshiro(12), 32, 48)
    c = analyze(sys, x)
    L = subband_layout(sys, size(x))
    @test L isa Vector{ShearletIndex{1}} && islowpass(L[1])
    @test nsubbands(sys, size(x)) == length(L) == size(c, 3)
    @test subband(sys, c, 2) == c[:, :, 2]
    @test parent(subband(sys, c, 2)) === c
    @test MultiscaleTransformsBase.signal_size(sys, c) == size(x)
    @test MultiscaleTransformsBase.supports_streaming(sys)
    s = similar_coefficients(sys, rand(Xoshiro(1), 0:9, 32, 48))
    @test size(s) == size(c) && eltype(s) == Float64
    ws = make_workspace(sys, Float32, size(x); flags = Shearlets.FFTW.ESTIMATE)
    @test ws isa AbstractTransformWorkspace
    @test transform(ws) == sys && size(ws) == size(x) && eltype(ws) == Float32
    @test eltype(make_workspace(sys, rand(Xoshiro(2), 0:9, 16, 16); flags = Shearlets.FFTW.ESTIMATE)) == Float64
    # Any integer works as the FFTW flags (0 is `FFTW.MEASURE`).
    @test eltype(make_workspace(sys, Float64, (16, 16); flags = 0)) == Float64
    @test eltype(make_workspace(sys, zeros(16, 16); flags = Int(Shearlets.FFTW.ESTIMATE))) == Float64
    # Mismatched workspaces fail fast, with messages.
    @test_throws "built for element type Float32" analyze(sys, x; workspace = ws)
    @test_throws "built for data of size" analyze(sys, Float32.(x[:, 1:32]); workspace = ws)
    @test_throws "the workspace was built for" analyze(BandlimitedShearlets(), Float32.(x); workspace = ws)
    # A streamed band of the wrong size, and real coefficients for complex data,
    # are rejected before anything is written.
    let s1 = BandlimitedShearlets(scales = 1), x16 = randn(16, 16)
        c16 = analyze(s1, x16)
        for T in (Float64, ComplexF64)
            cT = T.(c16)
            @test_throws "returned a band of size (8, 16), expected (16, 16)" synthesize!(i -> cT[1:8, :, i], zeros(T, 16, 16), s1)
        end
        @test_throws "complex data need complex coefficients, got Float64" analyze!(zero(c16), s1, complex.(x16, x16))
        # Sizes of any integer type.
        d32 = (Int32(16), Int32(16))
        for sys in (s1, NSSTShearlets(direction_levels = [2]), CompactShearlets(scales = 1))
            @test subband_layout(sys, d32) == subband_layout(sys, (16, 16))
        end
        @test_throws "support 2-D and 3-D data" subband_layout(s1, (Int32(4), Int32(4), Int32(4), Int32(4)))
        @test size(make_workspace(s1, Float64, d32)) == (16, 16)
        # Host arrays that are not `Array`s (a range, a BitMatrix) stay on the host.
        r = reshape(1.0:256.0, 16, 16)
        b = x16 .> 0
        @test analyze(s1, r) ≈ analyze(s1, collect(r))
        @test analyze(s1, b) ≈ analyze(s1, Float64.(b))
        @test make_workspace(s1, b) isa Shearlets.ShearletWorkspace
    end
    # The streaming forms take `threaded`, and return what the contract says.
    @test analyze((i, b) -> nothing, sys, x; threaded = true) === nothing
    y = similar(x)
    @test synthesize!(i -> subband(sys, c, i), y, sys; threaded = true) === y
    @test y ≈ x
    @test isempty(Test.detect_ambiguities(Shearlets, MultiscaleTransformsBase))
end

@testitem "Workspace keywords: flags, tasks, tune, prewarm" tags = [:interface, :workspace] begin
    using Random, Test
    using Shearlets.FFTW: FFTW
    rng = Xoshiro(13)
    # Allocations of a call through a function, not at top level (globals).
    allocs_analyze!(c, ws, x) = @allocated analyze!(c, ws, x; threaded = false)
    for sys in (BandlimitedShearlets(scales = 2), NSSTShearlets(direction_levels = [2, 3]), CompactShearlets(scales = 2))
        dims = (32, 40)
        x = randn(rng, dims...)
        ref = make_workspace(sys, Float64, dims; flags = FFTW.ESTIMATE)
        c = analyze(ref, x)
        y = synthesize(ref, c)
        # `tasks` bounds the engine's tasks and their buffers, on both forms.
        for w in (
                make_workspace(sys, Float64, dims; flags = FFTW.ESTIMATE, tasks = 1),
                make_workspace(sys, x; flags = FFTW.ESTIMATE, tasks = 1),
            )
            @test length(w.engine.buffers) == 1
            @test analyze(w, x) == c
            @test synthesize(w, c) ≈ y
        end
        w2 = make_workspace(sys, x; flags = FFTW.ESTIMATE, tasks = 2)
        @test length(w2.engine.buffers) <= 2
        @test analyze(w2, x) == c
        Threads.nthreads() > 1 && @test length(ref.engine.buffers) > 1
        @test_throws "tasks must be at least 1" make_workspace(sys, Float64, dims; tasks = 0)
        @test_throws "tasks must be at least 1" make_workspace(sys, x; tasks = 0)
        # `flags` reaches the FFT plans (`FFTW.MEASURE` is 0; its plan comes
        # last, since its wisdom can change the later `ESTIMATE` plans' rounding).
        @test ref.engine.pfwd.flags & FFTW.ESTIMATE != 0
        # `tune` is accepted and ignored on the host.
        for w in (
                make_workspace(sys, Float64, dims; flags = FFTW.ESTIMATE, tune = true),
                make_workspace(sys, x; flags = FFTW.ESTIMATE, tune = true),
            )
            @test w.engine.batch === nothing
            @test analyze(w, x) ≈ c
        end
        # `prewarm`: the workspace gives the same results, and its first call
        # allocates nothing.
        for w in (
                make_workspace(sys, Float64, dims; flags = FFTW.ESTIMATE, prewarm = true),
                make_workspace(sys, x; flags = FFTW.ESTIMATE, tasks = 1, prewarm = true),
            )
            cw = similar(c)
            @test allocs_analyze!(cw, w, x) == 0
            @test cw ≈ c
            @test synthesize(w, c) ≈ y
        end
        wm = make_workspace(sys, x; tasks = 1)
        @test wm.engine.pfwd.flags & FFTW.ESTIMATE == 0
        # Unknown keywords throw.
        @test_throws MethodError make_workspace(sys, Float64, dims; threads = 2)
        @test_throws MethodError make_workspace(sys, x; threads = 2)
    end
end

@testitem "Systems: show prints a keyword call" tags = [:interface] begin
    using Random
    # The two-argument form parses back to an equal system.
    roundtrip(sys) = Core.eval(@__MODULE__, Meta.parse(repr(sys))) == sys
    systems = (
        BandlimitedShearlets(),
        BandlimitedShearlets(scales = 3, smooth = true, alpha = 0.5),
        NSSTShearlets(),
        NSSTShearlets(direction_levels = [2, 3], filter_size = 0),
        NSSTShearlets(direction_levels = [2], pyramid = FilterPair([0.25, 0.5, 0.25], [-0.125, 0.25, 0.75, 0.25, -0.125])),
        CompactShearlets(),
        CompactShearlets(scales = 2, shear_levels = [0, 2], qmf = [0.25, 0.5, 0.25]),
    )
    for sys in systems
        @test roundtrip(sys)
        @test sprint(show, MIME("text/plain"), sys) == repr(sys)
    end
    @test repr(BandlimitedShearlets()) == "BandlimitedShearlets(scales = 0, smooth = false, alpha = 1.0)"
    @test repr(NSSTShearlets()) == "NSSTShearlets(direction_levels = [3, 3, 4, 4], filter_size = [32, 32, 16, 16])"
    @test repr(CompactShearlets()) == "CompactShearlets(scales = 4, shear_levels = [1, 1, 2, 2])"

    # A large custom filter is summarized; the text/plain display adds its taps.
    fan = CompactShearlets().fan
    fan2 = (fan .+ reverse(fan)) ./ 2 .+ 1.0e-3
    sys = CompactShearlets(scales = 2, fan = fan2)
    @test repr(sys) == "CompactShearlets(scales = 2, shear_levels = [1, 1], fan = <9×9 Matrix{Float64}>)"
    full = sprint(show, MIME("text/plain"), sys)
    @test startswith(full, repr(sys) * "\n  fan = 9×9 Matrix{Float64}:")
    @test count(==('\n'), full) == 10

    # Workspace mismatch messages name the systems briefly.
    x = randn(Xoshiro(3), 64, 64)
    ws = make_workspace(CompactShearlets(scales = 2), x)
    err = try
        analyze(CompactShearlets(scales = 3), x; workspace = ws)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("CompactShearlets(scales = 2, shear_levels = [1, 1])", msg)
    @test length(msg) < 200
end

@testitem "README examples run" tags = [:interface] begin
    # Every Julia block of the README, in order, in one module, except the
    # installation and GPU blocks: an API change that breaks one fails here.
    readme = read(joinpath(@__DIR__, "..", "..", "README.md"), String)
    blocks = [m.captures[1] for m in eachmatch(r"```julia\n(.*?)```"s, readme)]
    runnable = filter(b -> !occursin("Pkg", b) && !occursin("CUDA", b), blocks)
    @test !isempty(runnable)
    mod = Module()
    for b in runnable
        @test (include_string(mod, b); true)
    end
end
