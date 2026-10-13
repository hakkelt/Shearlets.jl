@testitem "Meyer windows: partitions of unity" tags = [:windows] begin
    using Shearlets: _meyer_aux, _meyer_psi1, _meyer_psi2, _meyer_phi
    xs = range(-0.5, 1.5; length = 2001)
    @test all(x -> _meyer_aux(x) + _meyer_aux(1 - x) ≈ 1, xs)
    # Radial windows over the scales, plus the lowpass, cover every frequency.
    @test all(range(0, 40; length = 4001)) do ω
        abs(_meyer_phi(ω)^2 + sum(j -> _meyer_psi1(ω / 4^j)^2, 0:3) - 1) < 1.0e-12
    end
    # The shears of the angular bump tile the line.
    @test all(x -> abs(sum(k -> _meyer_psi2(x + k)^2, -5:5) - 1) < 1.0e-12, range(-3, 3; length = 1201))
end

@testitem "Band-limited shearlets: Parseval frame and perfect reconstruction" tags = [:bandlimited] begin
    using Random
    for smooth in (false, true), dims in ((64, 64), (63, 64), (65, 47), (128, 96), (7, 9))
        sys = BandlimitedShearlets(smooth = smooth)
        S = shearlet_spectra(sys, dims)
        @test size(S) == (dims..., shearlet_nsubbands(sys, dims))
        @test maximum(abs, sum(abs2, S; dims = 3) .- 1) < 1.0e-12
        x = randn(Xoshiro(42), dims...)
        c = shearlet_forward(x, sys)
        @test eltype(c) == Float64
        @test sum(abs2, c) ≈ sum(abs2, x) rtol = 1.0e-12
        @test shearlet_inverse(c, sys) ≈ x atol = 1.0e-12
    end
end

@testitem "Band-limited shearlets: the masked FFT the spectra describe" tags = [:bandlimited] begin
    using FFTW, Random
    sys = BandlimitedShearlets(scales = 3)
    dims = (64, 48)
    x = randn(Xoshiro(42), dims...)
    S = shearlet_spectra(sys, dims)
    dense = ifft(S .* fft(x, (1, 2)), (1, 2))
    @test maximum(abs ∘ imag, dense) < 1.0e-14   # point-symmetric masks: real subbands
    @test shearlet_forward(x, sys) ≈ real.(dense) atol = 1.0e-13
end

@testitem "Band-limited shearlets: layout" tags = [:bandlimited] begin
    sys = BandlimitedShearlets(scales = 3)
    L = shearlet_layout(sys, (128, 128))
    @test length(L) == 2^(3 + 2) - 3            # 1 + Σⱼ 2^(j+1)
    @test islowpass(L[1]) && count(islowpass, L) == 1
    for j in 1:3
        @test count(i -> i.scale == j, L) == 2^(j + 1)
        @test count(i -> i.scale == j && i.cone == 1, L) == 2^j + 1   # seam shearlets listed here
    end
    @test issorted(L; by = i -> i.scale)
    @test shearlet_nsubbands(BandlimitedShearlets(), (256, 256)) == 61   # J = 4 by default
    @test_throws ArgumentError BandlimitedShearlets(scales = -1)
    @test_throws ArgumentError shearlet_layout(sys, (8, 8, 8, 8))
end

@testitem "Band-limited shearlets: a plane wave lands in its shear" tags = [:bandlimited] begin
    # Perfect reconstruction says nothing about geometry; check where energy goes.
    wave(dims, f) = [cospi(2 * (f[1] * (i - 1) / dims[1] + f[2] * (j - 1) / dims[2])) for i in 1:dims[1], j in 1:dims[2]]
    dims = (128, 128)
    for smooth in (false, true)
        sys = BandlimitedShearlets(scales = 3, smooth = smooth)
        L = shearlet_layout(sys, dims)
        # Scale 3 has 2^(3-1) = 4 shears per side; slope ω₂/ω₁ = -k/4 in cone 1.
        for (f, cone, k) in [
                ((40, 0), 1, 0), ((40, 10), 1, -1), ((40, 20), 1, -2), ((40, -20), 1, 2),
                ((40, 40), 1, -4), ((30, -30), 1, 4),
                ((0, 40), 2, 0), ((20, 40), 2, -2), ((-20, 40), 2, 2),
            ]
            c = shearlet_forward(wave(dims, f), sys)
            e = vec(sum(abs2, c; dims = (1, 2)))
            i = argmax(e)
            @test (L[i].scale, L[i].cone, L[i].shear[1]) == (3, cone, k)
            @test e[i] / sum(e) > 0.99
        end
    end
end

@testitem "Band-limited shearlets: shift invariance" tags = [:bandlimited] begin
    using Random
    sys = BandlimitedShearlets(scales = 3)
    x = randn(Xoshiro(42), 64, 80)
    s = (5, -9)
    @test maximum(abs, circshift(shearlet_forward(x, sys), (s..., 0)) - shearlet_forward(circshift(x, s), sys)) < 1.0e-10
end

@testitem "Band-limited shearlets: complex data and precision" tags = [:bandlimited] begin
    using Random
    sys = BandlimitedShearlets(scales = 2)
    rng = Xoshiro(42)
    a, b = randn(rng, 48, 64), randn(rng, 48, 64)
    z = complex.(a, b)
    cz = shearlet_forward(z, sys)
    @test eltype(cz) == ComplexF64
    @test cz == complex.(shearlet_forward(a, sys), shearlet_forward(b, sys))   # real-linear, bit for bit
    @test shearlet_inverse(cz, sys) ≈ z atol = 1.0e-12
    x32 = randn(rng, Float32, 48, 64)
    c32 = shearlet_forward(x32, sys)
    @test eltype(c32) == Float32
    @test maximum(abs, shearlet_inverse(c32, sys) - x32) < 1.0e-5
    # Integer input is transformed as floating point.
    @test eltype(shearlet_forward(rand(rng, 0:9, 16, 16), sys)) == Float64
end

@testitem "Band-limited shearlets: workspace, in-place and streaming forms" tags = [:bandlimited, :workspace] begin
    using Random
    sys = BandlimitedShearlets(scales = 3)
    dims = (96, 64)
    x = randn(Xoshiro(42), dims...)
    ws = make_shearlet_workspace(Float64, dims, sys)
    c = shearlet_forward(x, sys; workspace = ws)
    @test c == shearlet_forward(x, sys)
    for threading in (Disabled(), Enabled())
        c2 = similar(c)
        shearlet_forward!(c2, x, sys; workspace = ws, threading = threading)
        @test c2 == c
        y = similar(x)
        shearlet_inverse!(y, c, sys; workspace = ws, threading = threading)
        @test y ≈ x atol = 1.0e-12
    end
    # Streaming: the same subbands, one at a time.
    seen = Int[]
    shearlet_forward(x, sys; workspace = ws) do i, band
        push!(seen, i)
        @test band == selectdim(c, 3, i)
    end
    @test seen == 1:size(c, 3)
    @test shearlet_inverse(i -> selectdim(c, 3, i), sys, dims; workspace = ws) ≈ x atol = 1.0e-12
    # A view as the output, and as the input.
    big = zeros(dims..., size(c, 3) + 1)
    shearlet_forward!(view(big, :, :, 2:size(big, 3)), x, sys; workspace = ws)
    @test view(big, :, :, 2:size(big, 3)) == c
    @test shearlet_inverse(view(big, :, :, 2:size(big, 3)), sys) ≈ x atol = 1.0e-12
    # A workspace that does not fit is refused.
    @test_throws DimensionMismatch shearlet_forward(randn(64, 64), sys; workspace = ws)
    @test_throws ArgumentError shearlet_forward(x, BandlimitedShearlets(scales = 2); workspace = ws)
    @test_throws DimensionMismatch shearlet_forward!(zeros(dims..., 3), x, sys; workspace = ws)
    @test_throws ArgumentError shearlet_forward(Float32.(x), sys; workspace = ws)
    @test_throws ArgumentError shearlet_inverse!(similar(x), complex.(c), sys)
    # Views and transposes of host arrays need no workspace.
    @test shearlet_forward(view(hcat(x, x), :, 1:64), sys) ≈ shearlet_forward(x[:, 1:64], sys)
    @test shearlet_forward(transpose(x), sys) ≈ shearlet_forward(permutedims(x), sys)
    # Only 2-D and 3-D data.
    @test_throws ArgumentError shearlet_forward(randn(4, 4, 4, 4), BandlimitedShearlets(scales = 1))
end

@testitem "Band-limited shearlets: workspace paths do not allocate" tags = [:bandlimited, :workspace] begin
    using Random
    sys = BandlimitedShearlets(scales = 3)
    x = randn(Xoshiro(42), 64, 64)
    ws = make_shearlet_workspace(Float64, size(x), sys)
    c = shearlet_forward(x, sys; workspace = ws)
    y = similar(x)
    fwd(c, x, sys, ws) = @allocated shearlet_forward!(c, x, sys; workspace = ws, threading = Disabled())
    inv(y, c, sys, ws) = @allocated shearlet_inverse!(y, c, sys; workspace = ws, threading = Disabled())
    fwd(c, x, sys, ws), inv(y, c, sys, ws)
    @test fwd(c, x, sys, ws) == 0
    @test inv(y, c, sys, ws) == 0
end
