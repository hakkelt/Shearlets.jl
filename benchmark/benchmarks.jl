# Benchmark suite for Shearlets.jl (BenchmarkTools / AirspeedVelocity).
#
#     julia --project=benchmark -e 'include("benchmark/benchmarks.jl"); using BenchmarkTools;
#         BenchmarkTools.save(".temp/benchmark_results.json", run(SUITE; verbose = true))'

using BenchmarkTools, Random, Shearlets

const SUITE = BenchmarkGroup()

const CASES = [
    ("bandlimited", BandlimitedShearlets(scales = 4), (256, 256)),
    ("nsst", NSSTShearlets(directions = [3, 3, 4, 4]), (256, 256)),
    ("compact", CompactShearlets(scales = 4), (256, 256)),
    ("bandlimited3d", BandlimitedShearlets(scales = 2), (64, 64, 64)),
    ("compact3d", CompactShearlets(scales = 2), (64, 64, 64)),
]

for (name, sys, dims) in CASES, T in (Float32, ComplexF32)
    rng = Xoshiro(42)
    x = T <: Real ? randn(rng, Float32, dims...) : complex.(randn(rng, Float32, dims...), randn(rng, Float32, dims...))
    ws = make_shearlet_workspace(T, dims, sys)
    c = shearlet_forward(x, sys; workspace = ws)
    y = similar(x)
    g = SUITE[name][string(T)] = BenchmarkGroup()
    g["forward"] = @benchmarkable shearlet_forward!($c, $x, $sys; workspace = $ws)
    g["inverse"] = @benchmarkable shearlet_inverse!($y, $c, $sys; workspace = $ws)
    g["forward_serial"] = @benchmarkable shearlet_forward!($c, $x, $sys; workspace = $ws, threaded = false)
end
