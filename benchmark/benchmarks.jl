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
    ws = make_workspace(sys, T, dims)
    c = analyze(sys, x; workspace = ws)
    y = similar(x)
    g = SUITE[name][string(T)] = BenchmarkGroup()
    g["forward"] = @benchmarkable analyze!($c, $sys, $x; workspace = $ws)
    g["inverse"] = @benchmarkable synthesize!($y, $sys, $c; workspace = $ws)
    g["forward_serial"] = @benchmarkable analyze!($c, $sys, $x; workspace = $ws, threaded = false)
end
