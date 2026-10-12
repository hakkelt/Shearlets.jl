@testitem "Compact shearlets: filter building blocks" tags = [:compact] begin
    using Shearlets: _SHEARLAB_QMF, _maxflat_fan, _atrous_lowpass, _atrous_highpass, _upsample, _downsample, _conv
    using FFTW
    h = _SHEARLAB_QMF
    @test sum(h) ≈ 1
    @test h == reverse(h)
    # The 3 dB point of ShearLab's maximally flat lowpass sits at half the band.
    H(ω) = sum(h[n] * cos(ω * (n - 5)) for n in 1:9)
    @test H(π / 2)^2 ≈ 1 / 2 atol = 1.0e-12
    @test abs(H(π)) < 1.0e-12
    # The fan filter and its transpose split the plane: P + Pᵀ = 1.
    P = _maxflat_fan()
    @test size(P) == (9, 9)
    @test P + permutedims(P) ≈ [i == j == 5 ? 1.0 : 0.0 for i in 1:9, j in 1:9] atol = 1.0e-14
    # À trous filters: ĥ_j(0) = 1, ĝ_j(0) = 0, odd lengths.
    for j in 1:3
        @test sum(_atrous_lowpass(h, j)) ≈ 1
        @test abs(sum(_atrous_highpass(h, j))) < 1.0e-14
        @test isodd(length(_atrous_highpass(h, j)))
    end
    # Upsampling and downsampling keep the origin at the centre.
    a = collect(1.0:5.0)
    @test _upsample(a, 3, 1) == [1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5]
    @test _downsample(_upsample(a, 3, 1), 3, 1) == a
    @test _conv([1.0, 2.0, 1.0], [1.0]) ≈ [1, 2, 1]
end

@testitem "Compact shearlets: dual-frame perfect reconstruction" tags = [:compact] begin
    using Random
    for sys in (CompactShearlets(scales = 2), CompactShearlets(scales = 3, shear_levels = [0, 1, 2])), dims in ((64, 64), (63, 70))
        x = randn(Xoshiro(42), dims...)
        c = shearlet_forward(x, sys)
        @test size(c) == (dims..., shearlet_nsubbands(sys, dims))
        @test shearlet_inverse(c, sys) ≈ x atol = 1.0e-12
        A = shearlet_spectra(sys, dims)
        S = shearlet_spectra(sys, dims; synthesis = true)
        @test maximum(abs, sum(A .* S; dims = 3) .- 1) < 1.0e-12
        @test minimum(sum(abs2, A; dims = 3)) > 0.05   # a frame: no frequency is lost
    end
end

@testitem "Compact shearlets: layout" tags = [:compact] begin
    sys = CompactShearlets(scales = 4)
    @test sys.shear_levels == [1, 1, 2, 2]
    L = shearlet_layout(sys, (128, 128))
    @test length(L) == 1 + 2 * (5 + 5 + 9 + 9)
    @test count(i -> i.scale == 3 && i.cone == 2, L) == 9
    @test_throws ArgumentError CompactShearlets(scales = 0)
    @test_throws ArgumentError CompactShearlets(scales = 2, shear_levels = [1])
    @test_throws ArgumentError CompactShearlets(qmf = [0.5, 0.5])
    @test_throws ArgumentError CompactShearlets(fan = ones(2, 3))
    @test CompactShearlets() == CompactShearlets()
end

@testitem "Compact shearlets: a plane wave lands in its shear" tags = [:compact] begin
    wave(dims, f) = [cospi(2 * (f[1] * (i - 1) / dims[1] + f[2] * (j - 1) / dims[2])) for i in 1:dims[1], j in 1:dims[2]]
    dims = (128, 128)
    sys = CompactShearlets(scales = 3)   # shear level 2 at scale 3: slope -k/4
    ws = make_shearlet_workspace(Float64, dims, sys)
    L = shearlet_layout(sys, dims)
    for (f, cone, k) in [
            ((40, 0), 1, 0), ((40, 10), 1, -1), ((40, 20), 1, -2), ((40, -20), 1, 2),
            ((0, 40), 2, 0), ((20, 40), 2, -2), ((-20, 40), 2, 2),
        ]
        c = shearlet_forward(wave(dims, f), sys; workspace = ws)
        e = vec(sum(abs2, c; dims = (1, 2)))
        i = argmax(e)
        @test (L[i].scale, L[i].cone, L[i].shear[1]) == (3, cone, k)
        @test e[i] / sum(e) > 0.75
    end
end

@testitem "Compact shearlets: shift invariance and real-linearity" tags = [:compact] begin
    using Random
    sys = CompactShearlets(scales = 2)
    rng = Xoshiro(42)
    x, b = randn(rng, 48, 64), randn(rng, 48, 64)
    @test maximum(abs, circshift(shearlet_forward(x, sys), (4, 11, 0)) - shearlet_forward(circshift(x, (4, 11)), sys)) < 1.0e-10
    @test shearlet_forward(complex.(x, b), sys) == complex.(shearlet_forward(x, sys), shearlet_forward(b, sys))
end
