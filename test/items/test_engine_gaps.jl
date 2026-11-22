@testitem "Product masks: factors scalar along an axis" tags = [:workspace] begin
    using MultiscaleFilterBanks, Random
    rng = Xoshiro(5)
    n = (6, 5, 4)
    X = complex.(randn(rng, n...), randn(rng, n...))
    # Factors that are both a scalar along the first axis take the scalar-weight loop.
    f1 = rand(rng, 1, n[2], 1)
    f2 = rand(rng, 1, 1, n[3])
    m = Shearlets._ProductMask(f1, f2)
    ref = (f1 .* f2) .* X
    @test MultiscaleFilterBanks.mask_apply!(similar(X), m, X, 2.0) ≈ 2 .* ref
    # The weight defaults to one.
    @test MultiscaleFilterBanks.mask_apply!(similar(X), m, X) ≈ ref
    A = copy(X)
    MultiscaleFilterBanks.mask_accumulate!(A, m, X, 0.5)
    @test A ≈ X .+ 0.5 .* ref
    # Factors of the grid's length along the first axis.
    g1 = rand(rng, n...)
    g2 = rand(rng, 1, n[2], n[3])
    mg = Shearlets._ProductMask(g1, g2)
    @test MultiscaleFilterBanks.mask_apply!(similar(X), mg, X) ≈ (g1 .* g2) .* X
    # A factor that does not fit the grid is refused.
    bad = Shearlets._ProductMask(rand(rng, n[1] + 1, n[2], n[3]), g2)
    @test_throws "do not fit the grid" MultiscaleFilterBanks.mask_apply!(similar(X), bad, X)
end

@testitem "Engine helpers: device hooks without a device" tags = [:workspace] begin
    # Without the Adapt extension's method, moving a workspace is refused.
    @test_throws "needs the Adapt extension" Shearlets._to_device(nothing, zeros(2))
    # A shearlet engine has no separate lowpass.
    @test_throws "no separate lowpass" Shearlets._band_index(nothing)
    @test Shearlets._band_index((3,)) == 3
end

@testitem "Systems: layouts of any integer size type and bad sizes" tags = [:interface] begin
    for sys in (BandlimitedShearlets(scales = 2), CompactShearlets(scales = 2))
        @test subband_layout(sys, (Int32(16), Int32(16))) == subband_layout(sys, (16, 16))
        @test subband_layout(sys, (UInt8(16), 16)) == subband_layout(sys, (16, 16))
        @test_throws "support 2-D and 3-D data" subband_layout(sys, (16.0, 16.0))
        @test_throws "support 2-D and 3-D data" subband_layout(sys, ("a", "b"))
    end
end
