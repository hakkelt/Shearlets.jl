@testitem "Aqua quality checks" tags = [:quality] begin
    using Aqua
    Aqua.test_all(
        Shearlets;
        persistent_tasks = false,  # system resource check can fail in constrained envs
    )
end

@testitem "@test_call Shearlets public API" tags = [:jet, :quality] begin
    using JET, Random
    rng = Xoshiro(42)
    @test_call target_modules = (Shearlets,) BandlimitedShearlets(scales = 2)
    @test_call target_modules = (Shearlets,) NSSTShearlets(directions = [2, 3])
    @test_call target_modules = (Shearlets,) CompactShearlets(scales = 2)
    for (sys, dims) in (
            (BandlimitedShearlets(scales = 2), (32, 32)), (NSSTShearlets(directions = [2, 3]), (32, 32)),
            (CompactShearlets(scales = 1), (32, 32)), (BandlimitedShearlets(scales = 1), (8, 8, 8)),
            (CompactShearlets(scales = 1), (8, 8, 8)),
        )
        x = randn(rng, dims...)
        @test_call target_modules = (Shearlets,) shearlet_layout(sys, dims)
        @test_call target_modules = (Shearlets,) make_shearlet_workspace(Float64, dims, sys)
        ws = make_shearlet_workspace(Float64, dims, sys)
        c = shearlet_forward(x, sys; workspace = ws)
        y = similar(x)
        @test_opt target_modules = (Shearlets,) shearlet_forward!(c, x, sys; workspace = ws)
        @test_opt target_modules = (Shearlets,) shearlet_inverse!(y, c, sys; workspace = ws)
        @test_call target_modules = (Shearlets,) shearlet_forward(x, sys)
        @test_call target_modules = (Shearlets,) shearlet_inverse(c, sys)
    end
end
