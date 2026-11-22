# [How-to guides](@id howto)

Short recipes for common tasks. For worked experiments with figures, see the
[examples](@ref examples).

## Transform data repeatedly

`make_workspace` builds the masks, FFT plans and buffers for data of one size
and element type. The in-place forms then allocate nothing.

```jldoctest howto
julia> using Shearlets, Random

julia> Random.seed!(1); x = randn(64, 64);

julia> sys = BandlimitedShearlets(scales = 3);

julia> ws = make_workspace(sys, x);

julia> c = similar_coefficients(sys, x);

julia> analyze!(c, sys, x; workspace = ws);   # or analyze!(c, ws, x)

julia> y = similar(x); synthesize!(y, sys, c; workspace = ws);

julia> y ≈ x
true
```

## Pick out one subband

```jldoctest howto
julia> size(c), size(subband(sys, c, 5))
((64, 64, 29), (64, 64))
```

`subband_layout(sys, size(x))` lists what each index means.

## Transform complex data

Complex input is transformed as its real and imaginary parts, so the
coefficients are complex and the filters stay real.

```jldoctest howto
julia> z = x .+ im .* randn(64, 64);

julia> cz = analyze(sys, z); eltype(cz)
ComplexF64 (alias for Complex{Float64})

julia> z ≈ synthesize(sys, cz)
true
```

## Use several threads

The subbands are computed in parallel over Julia's threads. Start Julia with
`-t auto` (or another count), or pass `threaded = false` to run serially.

## Process data too large to hold every subband

`analyze(f, sys, x)` calls `f(i, band)` for each subband and does not keep them.
The calls may come from several threads at once.

## Run on a GPU

The transforms run on GPU backends with an FFT library: CUDA.jl (NVIDIA),
AMDGPU.jl (AMD) and Metal.jl (Apple silicon, `Float32` only); the test suite
runs on CUDA. OpenCL.jl is supported but much slower than the CPU, because it
has no FFT: FFTW runs the FFTs on the CPU on `CLArray`s in shared virtual memory
(OpenCL.jl's default on devices that support it). Build the workspace from
device data (the Adapt extension, which loads with any GPU package, moves it there):

```julia
using CUDA
xg = CuArray(x)
ws = make_workspace(sys, xg)
c = analyze(sys, xg; workspace = ws)
```

On the device the subbands are transformed in chunks, one batched FFT per chunk,
which keeps small grids from being bound by kernel launches; the chunk size
follows from the grid size. `make_workspace(sys, xg; tune = true)` measures the
fastest chunk size on the device instead, in a fraction of a second once compiled.
The chunks' stacks and batched FFT plans take up to about 80 MB of device
memory on top of the coefficients; grids too large to chunk, such as 128³, are
transformed one subband at a time.


## Find the right system

[Which system](@ref "Which system") in the explanation compares the three systems.
