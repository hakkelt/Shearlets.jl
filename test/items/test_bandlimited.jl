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
        @test size(S) == (dims..., nsubbands(sys, dims))
        @test maximum(abs, sum(abs2, S; dims = 3) .- 1) < 1.0e-12
        x = randn(Xoshiro(42), dims...)
        c = analyze(sys, x)
        @test eltype(c) == Float64
        @test sum(abs2, c) ≈ sum(abs2, x) rtol = 1.0e-12
        @test synthesize(sys, c) ≈ x atol = 1.0e-12
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
    @test analyze(sys, x) ≈ real.(dense) atol = 1.0e-13
end

@testitem "Band-limited shearlets: layout" tags = [:bandlimited] begin
    sys = BandlimitedShearlets(scales = 3)
    L = subband_layout(sys, (128, 128))
    @test length(L) == 2^(3 + 2) - 3            # 1 + Σⱼ 2^(j+1)
    @test islowpass(L[1]) && count(islowpass, L) == 1
    for j in 1:3
        @test count(i -> i.scale == j, L) == 2^(j + 1)
        @test count(i -> i.scale == j && i.cone == 1, L) == 2^j + 1   # seam shearlets listed here
    end
    @test issorted(L; by = i -> i.scale)
    @test nsubbands(BandlimitedShearlets(), (256, 256)) == 61   # J = 4 by default
    @test_throws ArgumentError BandlimitedShearlets(scales = -1)
    @test_throws ArgumentError subband_layout(sys, (8, 8, 8, 8))
end

@testitem "ShearletIndex: show round-trips" tags = [:bandlimited, :compact] begin
    for (sys, dims) in ((BandlimitedShearlets(scales = 2), (32, 32)), (BandlimitedShearlets(scales = 2), (16, 16, 16)), (CompactShearlets(scales = 2), (32, 32)))
        for i in subband_layout(sys, dims)
            s = repr(i)
            @test Core.eval(@__MODULE__, Meta.parse(s)) === i
            @test sprint(show, MIME("text/plain"), i) == s
        end
    end
    @test repr(subband_layout(BandlimitedShearlets(), (8, 8))[1]) == "ShearletIndex(scale = 0, cone = 0, shear = (0,))"
    @test ShearletIndex(scale = 2, cone = 1, shear = (-1, 3)) === ShearletIndex{2}(2, 1, (-1, 3))
    @test ShearletIndex(scale = 1, cone = 2, shear = [0]) === ShearletIndex{1}(1, 2, (0,))
end

@testitem "Band-limited shearlets: a plane wave lands in its shear" tags = [:bandlimited] begin
    # Perfect reconstruction says nothing about geometry; check where energy goes.
    wave(dims, f) = [cospi(2 * (f[1] * (i - 1) / dims[1] + f[2] * (j - 1) / dims[2])) for i in 1:dims[1], j in 1:dims[2]]
    dims = (128, 128)
    for smooth in (false, true)
        sys = BandlimitedShearlets(scales = 3, smooth = smooth)
        L = subband_layout(sys, dims)
        # Scale 3 has 2^(3-1) = 4 shears per side; slope ω₂/ω₁ = -k/4 in cone 1.
        for (f, cone, k) in [
                ((40, 0), 1, 0), ((40, 10), 1, -1), ((40, 20), 1, -2), ((40, -20), 1, 2),
                ((40, 40), 1, -4), ((30, -30), 1, 4),
                ((0, 40), 2, 0), ((20, 40), 2, -2), ((-20, 40), 2, 2),
            ]
            c = analyze(sys, wave(dims, f))
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
    @test maximum(abs, circshift(analyze(sys, x), (s..., 0)) - analyze(sys, circshift(x, s))) < 1.0e-10
end

@testitem "Band-limited shearlets: complex data and precision" tags = [:bandlimited] begin
    using Random
    sys = BandlimitedShearlets(scales = 2)
    rng = Xoshiro(42)
    a, b = randn(rng, 48, 64), randn(rng, 48, 64)
    z = complex.(a, b)
    cz = analyze(sys, z)
    @test eltype(cz) == ComplexF64
    @test cz == complex.(analyze(sys, a), analyze(sys, b))   # real-linear, bit for bit
    @test synthesize(sys, cz) ≈ z atol = 1.0e-12
    x32 = randn(rng, Float32, 48, 64)
    c32 = analyze(sys, x32)
    @test eltype(c32) == Float32
    @test maximum(abs, synthesize(sys, c32) - x32) < 1.0e-5
    # Integer input is transformed as floating point.
    @test eltype(analyze(sys, rand(rng, 0:9, 16, 16))) == Float64
end

@testitem "Band-limited shearlets: workspace, in-place and streaming forms" tags = [:bandlimited, :workspace] begin
    using Random
    sys = BandlimitedShearlets(scales = 3)
    dims = (96, 64)
    x = randn(Xoshiro(42), dims...)
    ws = make_workspace(sys, Float64, dims)
    c = analyze(sys, x; workspace = ws)
    @test c == analyze(sys, x)
    for threaded in (false, true)
        c2 = similar(c)
        analyze!(c2, sys, x; workspace = ws, threaded = threaded)
        @test c2 == c
        y = similar(x)
        synthesize!(y, sys, c; workspace = ws, threaded = threaded)
        @test y ≈ x atol = 1.0e-12
    end
    # Streaming: the same subbands, one at a time.
    seen = Int[]
    analyze(sys, x; workspace = ws) do i, band
        push!(seen, i)
        @test band == selectdim(c, 3, i)
    end
    @test seen == 1:size(c, 3)
    @test synthesize!(i -> selectdim(c, 3, i), similar(x), sys; workspace = ws) ≈ x atol = 1.0e-12
    # A view as the output, and as the input.
    big = zeros(dims..., size(c, 3) + 1)
    analyze!(view(big, :, :, 2:size(big, 3)), sys, x; workspace = ws)
    @test view(big, :, :, 2:size(big, 3)) == c
    @test synthesize(sys, view(big, :, :, 2:size(big, 3))) ≈ x atol = 1.0e-12
    # A workspace that does not fit is refused.
    @test_throws DimensionMismatch analyze(sys, randn(64, 64); workspace = ws)
    @test_throws ArgumentError analyze(BandlimitedShearlets(scales = 2), x; workspace = ws)
    @test_throws DimensionMismatch analyze!(zeros(dims..., 3), sys, x; workspace = ws)
    @test_throws ArgumentError analyze(sys, Float32.(x); workspace = ws)
    @test_throws ArgumentError synthesize!(similar(x), sys, complex.(c))
    # Views and transposes of host arrays need no workspace.
    @test analyze(sys, view(hcat(x, x), :, 1:64)) ≈ analyze(sys, x[:, 1:64])
    @test analyze(sys, transpose(x)) ≈ analyze(sys, permutedims(x))
    # Only 2-D and 3-D data.
    @test_throws ArgumentError analyze(BandlimitedShearlets(scales = 1), randn(4, 4, 4, 4))
end

@testitem "Band-limited shearlets: workspace paths do not allocate" tags = [:bandlimited, :workspace] begin
    using Random
    sys = BandlimitedShearlets(scales = 3)
    x = randn(Xoshiro(42), 64, 64)
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

@testitem "Arrays the FFT plans cannot take go through buffers" tags = [:workspace] begin
    using Random
    rng = Xoshiro(11)
    # (31, 33) Float32: the slices of stacked coefficients are not 16-byte
    # aligned, so the plans take any alignment.
    for (sys, dims) in ((BandlimitedShearlets(scales = 2), (31, 33)), (NSSTShearlets(direction_levels = [2, 2]), (32, 32))),
            T in (Float32, ComplexF32)
        x = T <: Real ? randn(rng, T, dims...) : complex.(randn(rng, Float32, dims...), randn(rng, Float32, dims...))
        ws = make_workspace(sys, T, dims)
        ref = analyze(sys, x; workspace = ws)
        n = size(ref, 3)
        # A strided view of the input, and of the coefficients.
        big = zeros(T, 2 .* dims)
        xv = view(big, 1:2:size(big, 1), 1:2:size(big, 2))
        xv .= x
        C = zeros(T, 2 * dims[1], dims[2], n)
        cv = view(C, 1:2:size(C, 1), :, :)
        @test analyze!(cv, sys, xv; workspace = ws) == ref
        # The staged synthesis runs serially, so its sums are added in another order.
        y = synthesize(sys, ref; workspace = ws)
        @test synthesize!(view(big, 2:2:size(big, 1), 1:2:size(big, 2)), sys, cv; workspace = ws) ≈ y
        # Other element types.
        c64 = analyze!(zeros(T <: Real ? Float64 : ComplexF64, dims..., n), sys, x; workspace = ws)
        @test c64 == ref
        @test synthesize!(similar(x), sys, c64; workspace = ws) ≈ y
        # Streaming with a strided band from `f`.
        z = synthesize!(i -> view(C, 1:2:size(C, 1), :, i), similar(x), sys; workspace = ws)
        @test z ≈ y
    end
end
