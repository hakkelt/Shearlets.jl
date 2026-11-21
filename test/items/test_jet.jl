@testitem "@test_call Shearlets public API" tags = [:jet] begin
    using JET, Random
    rng = Xoshiro(42)
    M = (Shearlets,)

    # ── Systems / indices ────────────────────────────────────────────────────
    @test_call target_modules = M BandlimitedShearlets(scales = 2)
    @test_call target_modules = M NSSTShearlets(direction_levels = [2, 3])
    @test_call target_modules = M CompactShearlets(scales = 2)
    @test_call target_modules = M ShearletIndex{1}(1, 1, (0,))
    @test_call target_modules = M islowpass(ShearletIndex{1}(0, 0, (0,)))

    systems = (
        (BandlimitedShearlets(scales = 2), (32, 32)), (NSSTShearlets(direction_levels = [2, 3]), (32, 32)),
        (CompactShearlets(scales = 1), (32, 32)), (BandlimitedShearlets(scales = 1), (8, 8, 8)),
        (CompactShearlets(scales = 1), (8, 8, 8)),
    )
    for (sys, dims) in systems
        x = randn(rng, dims...)
        @test_call target_modules = M subband_layout(sys, dims)
        @test_call target_modules = M nsubbands(sys, dims)
        @test_call target_modules = M shearlet_spectra(sys, dims)
        @test_call target_modules = M shearlet_spectra(sys, dims; T = Float32, synthesis = true)

        # ── Workspace + transforms ───────────────────────────────────────────
        @test_call target_modules = M make_workspace(sys, Float64, dims)
        @test_call target_modules = M make_workspace(sys, x)
        ws = make_workspace(sys, Float64, dims)
        @test_call target_modules = M analyze(sys, x)
        @test_call target_modules = M analyze(sys, x; workspace = ws)
        @test_call target_modules = M analyze(sys, x; threaded = false)
        c = analyze(sys, x; workspace = ws)
        y = similar(x)
        @test_call target_modules = M analyze!(c, sys, x; workspace = ws)
        @test_call target_modules = M synthesize(sys, c)
        @test_call target_modules = M synthesize(sys, c; workspace = ws, threaded = true)
        @test_call target_modules = M synthesize!(y, sys, c; workspace = ws)
        @test_call target_modules = M analyze(ws, x)
        @test_call target_modules = M analyze!(c, ws, x)
        @test_call target_modules = M synthesize(ws, c)
        @test_call target_modules = M synthesize!(y, ws, c)
        @test_call target_modules = M subband(sys, c, 2)
        @test_call target_modules = M similar_coefficients(sys, x)

        # ── Streaming forms ──────────────────────────────────────────────────
        @test_call target_modules = M analyze((i, band) -> nothing, sys, x; workspace = ws)
        @test_call target_modules = M synthesize!(i -> zeros(dims), similar(x), sys; workspace = ws)
        @test_call target_modules = M synthesize!(i -> zeros(Float32, dims), zeros(Float32, dims), sys)
    end

    # Complex and single-precision data take other element-type paths.
    sys = BandlimitedShearlets(scales = 2)
    xc = randn(rng, ComplexF64, 32, 32)
    @test_call target_modules = M analyze(sys, xc)
    cc = analyze(sys, xc)
    @test_call target_modules = M synthesize(sys, cc)
    x32 = randn(rng, Float32, 32, 32)
    @test_call target_modules = M analyze(sys, x32)
end

@testitem "@test_opt Shearlets public API" tags = [:jet] begin
    using JET, Random
    rng = Xoshiro(43)
    M = (Shearlets,)

    @test_opt target_modules = M BandlimitedShearlets(scales = 2)
    @test_opt target_modules = M NSSTShearlets(direction_levels = [2, 3])
    @test_opt target_modules = M CompactShearlets(scales = 2)
    @test_opt target_modules = M ShearletIndex{1}(1, 1, (0,))
    @test_opt target_modules = M islowpass(ShearletIndex{1}(0, 0, (0,)))

    for (sys, dims) in (
            (BandlimitedShearlets(scales = 2), (32, 32)), (NSSTShearlets(direction_levels = [2, 3]), (32, 32)),
            (CompactShearlets(scales = 1), (32, 32)), (BandlimitedShearlets(scales = 1), (8, 8, 8)),
            (CompactShearlets(scales = 1), (8, 8, 8)),
        )
        x = randn(rng, dims...)
        ws = make_workspace(sys, Float64, dims)
        c = analyze(sys, x; workspace = ws)
        y = similar(x)
        @test_opt target_modules = M subband_layout(sys, dims)
        @test_opt target_modules = M nsubbands(sys, dims)
        @test_opt target_modules = M analyze!(c, sys, x; workspace = ws)
        @test_opt target_modules = M synthesize!(y, sys, c; workspace = ws)
        @test_opt target_modules = M analyze!(c, ws, x)
        @test_opt target_modules = M synthesize!(y, ws, c)
        @test_opt target_modules = M analyze(sys, x; workspace = ws)
        @test_opt target_modules = M synthesize(sys, c; workspace = ws)
        @test_opt target_modules = M shearlet_spectra(sys, dims)
        @test_opt target_modules = M make_workspace(sys, Float64, dims)
        @test_opt target_modules = M analyze(sys, x)
        @test_opt target_modules = M synthesize(sys, c)
        @test_opt target_modules = M analyze((i, band) -> nothing, sys, x; workspace = ws)
        @test_opt target_modules = M synthesize!(i -> zeros(dims), similar(x), sys; workspace = ws)
    end
end
