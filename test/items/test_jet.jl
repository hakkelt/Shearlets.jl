@testitem "@test_call Shearlets public API" tags = [:jet] begin
    using JET, Random
    rng = Xoshiro(42)
    M = (Shearlets,)

    # ── Systems / indices ────────────────────────────────────────────────────
    @test_call target_modules = M BandlimitedShearlets(scales = 2)
    @test_call target_modules = M NSSTShearlets(directions = [2, 3])
    @test_call target_modules = M CompactShearlets(scales = 2)
    @test_call target_modules = M ShearletIndex{1}(1, 1, (0,))
    @test_call target_modules = M islowpass(ShearletIndex{1}(0, 0, (0,)))

    systems = (
        (BandlimitedShearlets(scales = 2), (32, 32)), (NSSTShearlets(directions = [2, 3]), (32, 32)),
        (CompactShearlets(scales = 1), (32, 32)), (BandlimitedShearlets(scales = 1), (8, 8, 8)),
        (CompactShearlets(scales = 1), (8, 8, 8)),
    )
    for (sys, dims) in systems
        x = randn(rng, dims...)
        @test_call target_modules = M shearlet_layout(sys, dims)
        @test_call target_modules = M shearlet_nsubbands(sys, dims)
        @test_call target_modules = M shearlet_spectra(sys, dims)
        @test_call target_modules = M shearlet_spectra(sys, dims; T = Float32, synthesis = true)

        # ── Workspace + transforms ───────────────────────────────────────────
        @test_call target_modules = M make_shearlet_workspace(Float64, dims, sys)
        ws = make_shearlet_workspace(Float64, dims, sys)
        @test_call target_modules = M shearlet_forward(x, sys)
        @test_call target_modules = M shearlet_forward(x, sys; workspace = ws)
        @test_call target_modules = M shearlet_forward(x, sys; threaded = false)
        c = shearlet_forward(x, sys; workspace = ws)
        y = similar(x)
        @test_call target_modules = M shearlet_forward!(c, x, sys; workspace = ws)
        @test_call target_modules = M shearlet_inverse(c, sys)
        @test_call target_modules = M shearlet_inverse(c, sys; workspace = ws, threaded = true)
        @test_call target_modules = M shearlet_inverse!(y, c, sys; workspace = ws)

        # ── Streaming forms ──────────────────────────────────────────────────
        @test_call target_modules = M shearlet_forward((i, band) -> nothing, x, sys; workspace = ws)
        @test_call target_modules = M shearlet_inverse(i -> zeros(dims), sys, dims; workspace = ws)
        @test_call target_modules = M shearlet_inverse(i -> zeros(Float32, dims), sys, dims; T = Float32)
    end

    # Complex and single-precision data take other element-type paths.
    sys = BandlimitedShearlets(scales = 2)
    xc = randn(rng, ComplexF64, 32, 32)
    @test_call target_modules = M shearlet_forward(xc, sys)
    cc = shearlet_forward(xc, sys)
    @test_call target_modules = M shearlet_inverse(cc, sys)
    x32 = randn(rng, Float32, 32, 32)
    @test_call target_modules = M shearlet_forward(x32, sys)
end

@testitem "@test_opt Shearlets public API" tags = [:jet] begin
    using JET, Random
    rng = Xoshiro(43)
    M = (Shearlets,)

    @test_opt target_modules = M BandlimitedShearlets(scales = 2)
    @test_opt target_modules = M NSSTShearlets(directions = [2, 3])
    @test_opt target_modules = M CompactShearlets(scales = 2)
    @test_opt target_modules = M ShearletIndex{1}(1, 1, (0,))
    @test_opt target_modules = M islowpass(ShearletIndex{1}(0, 0, (0,)))

    for (sys, dims) in (
            (BandlimitedShearlets(scales = 2), (32, 32)), (NSSTShearlets(directions = [2, 3]), (32, 32)),
            (CompactShearlets(scales = 1), (32, 32)), (BandlimitedShearlets(scales = 1), (8, 8, 8)),
            (CompactShearlets(scales = 1), (8, 8, 8)),
        )
        x = randn(rng, dims...)
        ws = make_shearlet_workspace(Float64, dims, sys)
        c = shearlet_forward(x, sys; workspace = ws)
        y = similar(x)
        @test_opt target_modules = M shearlet_layout(sys, dims)
        @test_opt target_modules = M shearlet_nsubbands(sys, dims)
        @test_opt target_modules = M shearlet_forward!(c, x, sys; workspace = ws)
        @test_opt target_modules = M shearlet_inverse!(y, c, sys; workspace = ws)
        @test_opt target_modules = M shearlet_forward(x, sys; workspace = ws)
        @test_opt target_modules = M shearlet_inverse(c, sys; workspace = ws)
        @test_opt target_modules = M shearlet_spectra(sys, dims)
        @test_opt target_modules = M make_shearlet_workspace(Float64, dims, sys)
        @test_opt target_modules = M shearlet_forward(x, sys)
        @test_opt target_modules = M shearlet_inverse(c, sys)
        @test_opt target_modules = M shearlet_forward((i, band) -> nothing, x, sys; workspace = ws)
        @test_opt target_modules = M shearlet_inverse(i -> zeros(dims), sys, dims; workspace = ws)
    end
end
