@testitem "NSST: perfect reconstruction" tags = [:nsst] begin
    using Random
    for sys in (
                NSSTShearlets(direction_levels = [2, 3, 3]),
                NSSTShearlets(direction_levels = [2, 3], filter_size = 0),
                NSSTShearlets(direction_levels = [3], filter_size = 9),
            ), dims in ((64, 64), (63, 80))
        x = randn(Xoshiro(42), dims...)
        c = analyze(sys, x)
        @test size(c) == (dims..., nsubbands(sys, dims))
        @test eltype(c) == Float64
        @test synthesize(sys, c) ≈ x atol = 1.0e-12
    end
end

@testitem "NSST: shearing filters sum to one, and the pyramid telescopes" tags = [:nsst] begin
    sys = NSSTShearlets(direction_levels = [2, 3])
    dims = (48, 64)
    A = shearlet_spectra(sys, dims)
    S = shearlet_spectra(sys, dims; synthesis = true)
    @test maximum(abs, sum(A .* S; dims = 3) .- 1) < 1.0e-12
    L = subband_layout(sys, dims)
    # The directional subbands of a scale share their synthesis mask.
    for j in 1:2
        idx = findall(i -> i.scale == j, L)
        @test length(idx) == 2^sys.direction_levels[j]
        @test all(i -> selectdim(S, 3, i) == selectdim(S, 3, idx[1]), idx)
    end
end

@testitem "NSST: a plane wave lands in its shear" tags = [:nsst] begin
    wave(dims, f) = [cospi(2 * (f[1] * (i - 1) / dims[1] + f[2] * (j - 1) / dims[2])) for i in 1:dims[1], j in 1:dims[2]]
    dims = (128, 128)
    for fs in (nothing, 0)
        sys = NSSTShearlets(direction_levels = [2, 3, 3], filter_size = fs)
        L = subband_layout(sys, dims)
        # 2^3 directions: s = 2 shears per side, slope ω₂/ω₁ = -k/2 in cone 1.
        for (f, cone, k) in [
                ((40, 0), 1, 0), ((40, 20), 1, -1), ((40, -20), 1, 1), ((40, 40), 1, -2),
                ((0, 40), 2, 0), ((20, 40), 2, -1), ((-20, 40), 2, 1),
            ]
            c = analyze(sys, wave(dims, f))
            e = vec(sum(abs2, c; dims = (1, 2)))
            # The pyramid spreads a frequency over neighboring scales; take the
            # direction that holds most energy, summed over scales.
            dirs = unique((i.cone, i.shear) for i in L if !islowpass(i) && i.scale >= 2)
            best = argmax(d -> sum(e[i] for i in eachindex(L) if (L[i].cone, L[i].shear) == d && L[i].scale >= 2), dirs)
            @test best == (cone, (k,))
        end
    end
end

@testitem "NSST: shift invariance and real-linearity" tags = [:nsst] begin
    using Random
    sys = NSSTShearlets(direction_levels = [2, 3])
    rng = Xoshiro(42)
    x = randn(rng, 64, 48)
    @test maximum(abs, circshift(analyze(sys, x), (3, -7, 0)) - analyze(sys, circshift(x, (3, -7)))) < 1.0e-10
    b = randn(rng, 64, 48)
    @test analyze(sys, complex.(x, b)) == complex.(analyze(sys, x), analyze(sys, b))
end

@testitem "NSST: argument checks" tags = [:nsst] begin
    @test_throws ArgumentError NSSTShearlets(direction_levels = Int[])
    @test_throws ArgumentError NSSTShearlets(direction_levels = [1, 3])
    @test_throws ArgumentError NSSTShearlets(direction_levels = [2, 3], filter_size = [16])
    @test_throws ArgumentError NSSTShearlets(pyramid = FilterPair([0.5, 0.5], [1.0, 1.0]))
    @test_throws "pyramid.g must be symmetric" NSSTShearlets(pyramid = FilterPair([0.25, 0.5, 0.25], [0.5, 1.0, 0.6]))
    h = collect(CDF97.h)
    h[1] += 1.0e-9
    s = NSSTShearlets(direction_levels = [2], pyramid = FilterPair(h, collect(CDF97.g)))
    @test s.pyramid.h == reverse(s.pyramid.h)
    x = randn(32, 32)
    @test synthesize(s, analyze(s, x)) ≈ x
    @test NSSTShearlets(direction_levels = [2, 3, 4, 4]).filter_size == [32, 32, 16, 16]
    @test NSSTShearlets() == NSSTShearlets()
    @test hash(NSSTShearlets()) == hash(NSSTShearlets())
    other = FilterPair([0.25, 0.5, 0.25], [-0.125, -0.25, 0.75, -0.25, -0.125])
    @test NSSTShearlets(pyramid = other) == NSSTShearlets(pyramid = other)
    @test hash(NSSTShearlets(pyramid = other)) == hash(NSSTShearlets(pyramid = other))
    @test NSSTShearlets(pyramid = other) != NSSTShearlets()
    @test hash(CompactShearlets(scales = 1)) == hash(CompactShearlets(scales = 1))
    @test_throws ArgumentError subband_layout(NSSTShearlets(), (8, 8, 8))
    @test_throws ArgumentError analyze(NSSTShearlets(), randn(8, 8, 8))
    @test_throws ArgumentError shearlet_spectra(NSSTShearlets(), (8, 8, 8))
end
