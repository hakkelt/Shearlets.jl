@testitem "3-D band-limited shearlets: Parseval frame, perfect reconstruction, seams" tags = [:bandlimited, :threed] begin
    using Random
    for smooth in (false, true), dims in ((32, 32, 32), (24, 32, 20))
        sys = BandlimitedShearlets(scales = 2, smooth = smooth)
        L = subband_layout(sys, dims)
        # Glued seams: one shearlet per antipodal pair of points on the surface of
        # the cube [-s, s]³ — 13 for s = 1, 49 for s = 2.
        @test [count(i -> i.scale == j, L) for j in 1:2] == [13, 49]
        x = randn(Xoshiro(42), dims...)
        c = analyze(sys, x)
        @test size(c) == (dims..., length(L))
        @test sum(abs2, c) ≈ sum(abs2, x) rtol = 1.0e-12
        @test synthesize(sys, c) ≈ x atol = 1.0e-12
    end
end

@testitem "3-D compact shearlets: dual-frame perfect reconstruction" tags = [:compact, :threed] begin
    using Random
    sys = CompactShearlets(scales = 2)
    dims = (24, 32, 20)
    L = subband_layout(sys, dims)
    @test length(L) == 1 + 3 * (5^2 + 5^2)
    x = randn(Xoshiro(42), dims...)
    c = analyze(sys, x)
    @test synthesize(sys, c) ≈ x atol = 1.0e-12
    A = shearlet_spectra(sys, dims)
    S = shearlet_spectra(sys, dims; synthesis = true)
    @test maximum(abs, sum(A .* S; dims = 4) .- 1) < 1.0e-12
end

@testitem "3-D shearlets: a plane wave lands in its pyramid and shears" tags = [:bandlimited, :compact, :threed] begin
    wave(dims, f) = [cospi(2 * (f[1] * (i - 1) / dims[1] + f[2] * (j - 1) / dims[2] + f[3] * (k - 1) / dims[3])) for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]]
    dims = (32, 32, 32)
    # Scale 2 has s = 2 shears per side; shear k_e sits at ω_e / ω_c = -k_e / s.
    for sys in (BandlimitedShearlets(scales = 2), CompactShearlets(scales = 2))
        ws = make_workspace(sys, Float64, dims)
        L = subband_layout(sys, dims)
        for (f, cone, k) in [
                ((12, 0, 0), 1, (0, 0)), ((12, 6, 0), 1, (-1, 0)), ((12, 0, -6), 1, (0, 1)),
                ((0, 12, 6), 2, (0, -1)), ((6, 0, 12), 3, (-1, 0)), ((-6, 6, 12), 3, (1, -1)),
            ]
            c = analyze(sys, wave(dims, f); workspace = ws)
            e = vec(sum(abs2, c; dims = (1, 2, 3)))
            i = argmax(e)
            @test (L[i].scale, L[i].cone, L[i].shear) == (2, cone, k)
        end
    end
end

@testitem "3-D shearlets: shift invariance and real-linearity" tags = [:threed] begin
    using Random
    rng = Xoshiro(42)
    x, b = randn(rng, 16, 20, 24), randn(rng, 16, 20, 24)
    for sys in (BandlimitedShearlets(scales = 1), CompactShearlets(scales = 1))
        c = analyze(sys, x)
        @test maximum(abs, circshift(c, (3, -5, 7, 0)) - analyze(sys, circshift(x, (3, -5, 7)))) < 1.0e-10
        @test analyze(sys, complex.(x, b)) == complex.(c, analyze(sys, b))
    end
end

@testitem "3-D shearlets: workspace paths do not allocate" tags = [:threed, :workspace] begin
    using Random
    x = randn(Xoshiro(42), 16, 16, 16)
    for sys in (BandlimitedShearlets(scales = 1), CompactShearlets(scales = 1))
        ws = make_workspace(sys, Float64, size(x))
        c = analyze(sys, x; workspace = ws)
        y = similar(x)
        fwd(c, x, sys, ws) = @allocated analyze!(c, sys, x; workspace = ws, threaded = false)
        inv(y, c, sys, ws) = @allocated synthesize!(y, sys, c; workspace = ws, threaded = false)
        fwd_ws(c, x, ws) = @allocated analyze!(c, ws, x; threaded = false)
        inv_ws(y, c, ws) = @allocated synthesize!(y, ws, c; threaded = false)
        fwd(c, x, sys, ws), inv(y, c, sys, ws), fwd_ws(c, x, ws), inv_ws(y, c, ws)
        @test fwd(c, x, sys, ws) == 0
        @test inv(y, c, sys, ws) == 0
        @test fwd_ws(c, x, ws) == 0
        @test inv_ws(y, c, ws) == 0
    end
end

@testitem "α-shearlets: Parseval for every anisotropy" tags = [:bandlimited] begin
    using Random
    dims = (64, 64)
    x = randn(Xoshiro(42), dims...)
    counts = Int[]
    for α in (0.0, 0.5, 1.0, 1.5, 2.0)
        sys = BandlimitedShearlets(scales = 3, alpha = α)
        c = analyze(sys, x)
        push!(counts, size(c, 3))
        @test sum(abs2, c) ≈ sum(abs2, x) rtol = 1.0e-12
        @test synthesize(sys, c) ≈ x atol = 1.0e-12
    end
    # Smaller α: more shears at fine scales.
    @test issorted(counts; rev = true)
    @test counts[3] == nsubbands(BandlimitedShearlets(scales = 3), dims)
    @test_throws ArgumentError BandlimitedShearlets(alpha = 2.5)
end
