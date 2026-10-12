@testitem "NSST: perfect reconstruction" tags = [:nsst] begin
    using Random
    for sys in (
                NSSTShearlets(directions = [2, 3, 3]),
                NSSTShearlets(directions = [2, 3], filter_size = 0),
                NSSTShearlets(directions = [3], filter_size = 9),
            ), dims in ((64, 64), (63, 80))
        x = randn(Xoshiro(42), dims...)
        c = shearlet_forward(x, sys)
        @test size(c) == (dims..., shearlet_nsubbands(sys, dims))
        @test eltype(c) == Float64
        @test shearlet_inverse(c, sys) ≈ x atol = 1.0e-12
    end
end

@testitem "NSST: shearing filters sum to one, and the pyramid telescopes" tags = [:nsst] begin
    sys = NSSTShearlets(directions = [2, 3])
    dims = (48, 64)
    A = shearlet_spectra(sys, dims)
    S = shearlet_spectra(sys, dims; synthesis = true)
    @test maximum(abs, sum(A .* S; dims = 3) .- 1) < 1.0e-12
    L = shearlet_layout(sys, dims)
    # The directional subbands of a scale share their synthesis mask.
    for j in 1:2
        idx = findall(i -> i.scale == j, L)
        @test length(idx) == 2^sys.directions[j]
        @test all(i -> selectdim(S, 3, i) == selectdim(S, 3, idx[1]), idx)
    end
end

@testitem "NSST: a plane wave lands in its shear" tags = [:nsst] begin
    wave(dims, f) = [cospi(2 * (f[1] * (i - 1) / dims[1] + f[2] * (j - 1) / dims[2])) for i in 1:dims[1], j in 1:dims[2]]
    dims = (128, 128)
    for fs in (nothing, 0)
        sys = NSSTShearlets(directions = [2, 3, 3], filter_size = fs)
        L = shearlet_layout(sys, dims)
        # 2^3 directions: s = 2 shears per side, slope ω₂/ω₁ = -k/2 in cone 1.
        for (f, cone, k) in [
                ((40, 0), 1, 0), ((40, 20), 1, -1), ((40, -20), 1, 1), ((40, 40), 1, -2),
                ((0, 40), 2, 0), ((20, 40), 2, -1), ((-20, 40), 2, 1),
            ]
            c = shearlet_forward(wave(dims, f), sys)
            e = vec(sum(abs2, c; dims = (1, 2)))
            # The pyramid spreads a frequency over neighbouring scales; take the
            # direction that holds most energy, summed over scales.
            dirs = unique((i.cone, i.shear) for i in L if !islowpass(i) && i.scale >= 2)
            best = argmax(d -> sum(e[i] for i in eachindex(L) if (L[i].cone, L[i].shear) == d && L[i].scale >= 2), dirs)
            @test best == (cone, (k,))
        end
    end
end

@testitem "NSST: shift invariance and real-linearity" tags = [:nsst] begin
    using Random
    sys = NSSTShearlets(directions = [2, 3])
    rng = Xoshiro(42)
    x = randn(rng, 64, 48)
    @test maximum(abs, circshift(shearlet_forward(x, sys), (3, -7, 0)) - shearlet_forward(circshift(x, (3, -7)), sys)) < 1.0e-10
    b = randn(rng, 64, 48)
    @test shearlet_forward(complex.(x, b), sys) == complex.(shearlet_forward(x, sys), shearlet_forward(b, sys))
end

@testitem "NSST: argument checks" tags = [:nsst] begin
    @test_throws ArgumentError NSSTShearlets(directions = Int[])
    @test_throws ArgumentError NSSTShearlets(directions = [1, 3])
    @test_throws ArgumentError NSSTShearlets(directions = [2, 3], filter_size = [16])
    @test_throws ArgumentError NSSTShearlets(pyramid = FilterPair([0.5, 0.5], [1.0, 1.0]))
    @test NSSTShearlets(directions = [2, 3, 4, 4]).filter_size == [32, 32, 16, 16]
    @test NSSTShearlets() == NSSTShearlets()
    @test_throws ArgumentError shearlet_layout(NSSTShearlets(), (8, 8, 8))
end
