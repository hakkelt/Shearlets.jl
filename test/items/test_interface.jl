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
    # The streaming forms take `threaded`, and return what the contract says.
    @test analyze((i, b) -> nothing, sys, x; threaded = true) === nothing
    y = similar(x)
    @test synthesize!(i -> subband(sys, c, i), y, sys; threaded = true) === y
    @test y ≈ x
    @test isempty(Test.detect_ambiguities(Shearlets, MultiscaleTransformsBase))
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
