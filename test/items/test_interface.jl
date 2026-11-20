@testitem "MultiscaleTransformsBase conformance" tags = [:interface] begin
    using MultiscaleTransformsBase.Testing: test_transform
    using Random
    rng = Xoshiro(11)
    for (sys, dims, other) in (
            (BandlimitedShearlets(scales = 2), (32, 40), (32, 32)),
            (NSSTShearlets(directions = [2, 3]), (32, 40), (32, 32)),
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
