# [Common tasks](@id howto)

The examples on this page use one small image and the band-limited system:

```jldoctest howto
julia> using Shearlets, Random

julia> Random.seed!(1); x = randn(64, 64);

julia> sys = BandlimitedShearlets(scales = 3);
```

`analyze(sys, x)` returns the coefficients: one array of size
`(size(x)..., n)` with one subband per slice of the last axis, in the order
of `subband_layout(sys, size(x))`. `synthesize(sys, c)` inverts it.

## Reuse a workspace for repeated transforms

Without a workspace, every call builds the system's frequency masks and FFT
plans for the data size. `make_workspace` builds them once for data like `x`
(same size and element type):

```jldoctest howto
julia> ws = make_workspace(sys, x);

julia> c = similar_coefficients(sys, x);

julia> analyze!(c, sys, x; workspace = ws);   # or analyze!(c, ws, x)

julia> y = similar(x); synthesize!(y, sys, c; workspace = ws);

julia> y ≈ x
true
```

With a workspace and `threaded = false`, `analyze!` and `synthesize!`
allocate nothing. A workspace holds scratch buffers, so do not use one in two
transforms running at the same time; build one per task instead.

## Pick out one subband

`subband(sys, c, i)` is a view of subband `i`; `subband_layout` says which
scale, cone and shear it has (see [`ShearletIndex`](@ref)):

```jldoctest howto
julia> size(c), size(subband(sys, c, 5))
((64, 64, 29), (64, 64))

julia> subband_layout(sys, size(x))[5]
ShearletIndex(scale = 1, cone = 2, shear = (0,))
```

## Transform complex data

Complex input is transformed as its real and imaginary parts, so the
coefficients are complex:

```jldoctest howto
julia> z = x .+ im .* randn(64, 64);

julia> cz = analyze(sys, z); eltype(cz)
ComplexF64 (alias for Complex{Float64})

julia> z ≈ synthesize(sys, cz)
true
```

## Use several threads

`analyze`, `synthesize` and their `!` forms compute the subbands in
parallel over Julia's threads. Start Julia with `--threads` (for example
`julia --threads=auto`), or pass `threaded = false` to run on one thread.
`make_workspace(sys, x; tasks = k)` limits a workspace to `k` tasks.

## Process data too large to hold every subband

The streaming forms handle one subband at a time, so the full coefficient
array is never held. `analyze(f, sys, x)` calls `f(i, band)` for every
subband `i`, in `subband_layout` order, on the calling task. `band` is a
buffer that the next call overwrites: copy or reduce it inside `f`. The
subbands are computed one after another, so `threaded` has no effect here.

```jldoctest howto
julia> energy = zeros(nsubbands(sys, size(x)));

julia> analyze(sys, x) do i, band
           energy[i] = sum(abs2, band)
       end

julia> sum(energy) ≈ sum(abs2, x)   # the band-limited system keeps the energy
true
```

`synthesize!(f, y, sys)` is the inverse: it calls `f(i)` for every subband
in the same order, and `f` returns subband `i` (it may reuse one buffer):

```jldoctest howto
julia> synthesize!(i -> subband(sys, c, i), y, sys);

julia> y ≈ x
true
```

## Run on a GPU

The transforms run on device arrays whose package provides FFT plans, such as
CUDA.jl and AMDGPU.jl. Build the workspace from device data; the Adapt
extension, which loads with the GPU package, moves the masks to the device.
Device data always needs a workspace:

```julia
using Shearlets, CUDA
sys = BandlimitedShearlets(scales = 3)
xg = CUDA.randn(Float32, 256, 256)
ws = make_workspace(sys, xg)
c = analyze(sys, xg; workspace = ws)
xr = synthesize(sys, c; workspace = ws)
```

On a device, several subbands share one batched FFT. `make_workspace(sys, xg; tune = true)`
times several batch sizes and keeps the fastest; this helps mostly for
`NSSTShearlets`.
