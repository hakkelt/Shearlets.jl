@testitem "Generic axes: transforms" tags = [:quality] begin
    using OffsetArrays, Random
    msg = "offset arrays are not supported"
    rng = Xoshiro(21)
    for (sys, dims) in (
                (BandlimitedShearlets(scales = 2), (16, 16)),
                (NSSTShearlets(direction_levels = [2, 3]), (16, 16)),
                (CompactShearlets(scales = 2), (17, 16)),
                (BandlimitedShearlets(scales = 1), (8, 8, 8)),
            ), T in (Float64, ComplexF64)
        x = randn(rng, T, dims...)
        c = analyze(sys, x)
        xo = OffsetArray(x, (0, 3, ntuple(_ -> 0, length(dims) - 2)...))
        co = OffsetArray(c, (2, ntuple(_ -> 0, ndims(c) - 1)...))
        # Offset arrays are rejected by every entry point, in every argument.
        @test_throws msg analyze(sys, xo)
        @test_throws msg analyze(sys, xo; threaded = false)
        @test_throws msg analyze!(copy(c), sys, xo)
        @test_throws msg analyze!(co, sys, x)
        @test_throws msg analyze((i, b) -> nothing, sys, xo)
        @test_throws msg synthesize(sys, co)
        @test_throws msg synthesize!(similar(x), sys, co)
        @test_throws msg synthesize!(xo, sys, c)
        @test_throws msg synthesize!(i -> selectdim(c, ndims(c), i), xo, sys)
        @test_throws msg synthesize!(i -> OffsetArray(selectdim(c, ndims(c), i), ntuple(_ -> 1, ndims(c) - 1)...), similar(x), sys)
        @test_throws msg similar_coefficients(sys, xo)
        @test_throws msg make_workspace(sys, xo)

        # Views give the results of the arrays they wrap.
        xv = view(parent(similar(x, dims .+ 1)), axes(x)...)
        copyto!(xv, x)
        @test analyze(sys, xv) == c
        cv = view(analyze(sys, x), axes(c)...)
        @test analyze!(similar(c), sys, xv) == c
        @test synthesize(sys, cv) == synthesize(sys, c)
        yv = view(parent(similar(x, dims .+ 1)), axes(x)...)
        @test synthesize!(yv, sys, cv) == synthesize(sys, c)
        @test similar_coefficients(sys, xv) isa AbstractArray
        @test size(similar_coefficients(sys, xv)) == size(c)
        bands = Any[]
        analyze((i, b) -> push!(bands, copy(b)), sys, xv)
        @test cat(bands...; dims = ndims(c)) ≈ c rtol = 1.0e-12
        y = similar(x)
        synthesize!(i -> selectdim(c, ndims(c), i), y, sys)
        @test y ≈ synthesize(sys, c) rtol = 1.0e-12
    end
end

@testitem "Generic axes: system constructors" tags = [:quality] begin
    using OffsetArrays
    msg = "offset arrays are not supported"
    qmf = [0.25, 0.5, 0.25]
    fan = [0.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 0.0] ./ 8
    @test_throws msg CompactShearlets(scales = 1, qmf = OffsetArray(qmf, 0:2))
    @test_throws msg CompactShearlets(scales = 1, fan = OffsetArray(fan, 0:2, 0:2))
    sv = CompactShearlets(scales = 1, qmf = view(vcat(qmf, 0.0), 1:3), fan = view(fan, :, :))
    sp = CompactShearlets(scales = 1; qmf, fan)
    @test sv.qmf == sp.qmf && sv.fan == sp.fan
end
