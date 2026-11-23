# API Reference

Every type and function of Shearlets, including the names it re-exports from
MultiscaleTransformsBase and MultiscaleFilterBanks.

```@meta
CurrentModule = Shearlets
```

## Module

```@docs
Shearlets
MultiscaleFilterBanks.MultiscaleFilterBanks
```

## Shearlet systems

```@docs
ShearletSystem
BandlimitedShearlets
NSSTShearlets
CompactShearlets
ShearletIndex
islowpass
subband_layout(::ShearletSystem, ::Tuple)
shearlet_spectra
```

## Transforms

The verbs of
[MultiscaleTransformsBase](https://github.com/hakkelt/MultiscaleTransformsBase.jl),
for shearlet systems.  Every form also takes a workspace in place of the
system: `analyze(ws, x)`, `analyze!(c, ws, x)`, `synthesize(ws, c)`,
`synthesize!(x, ws, c)`, `analyze(f, ws, x)` and `synthesize!(f, x, ws)`.

```@docs
analyze(::ShearletSystem, ::AbstractArray{<:Number})
analyze!(::AbstractArray{<:Number}, ::ShearletSystem, ::AbstractArray{<:Number})
synthesize(::ShearletSystem, ::AbstractArray{<:Number})
synthesize!(::AbstractArray{<:Number}, ::ShearletSystem, ::AbstractArray{<:Number})
analyze(::Any, ::ShearletSystem, ::AbstractArray{<:Number})
synthesize!(::Any, ::AbstractArray{<:Number}, ::ShearletSystem)
ShearletWorkspace
make_workspace(::ShearletSystem, ::Type{T}, ::Dims) where {T <: Number}
make_workspace(::ShearletSystem, ::AbstractArray)
```

## Coefficients

```@docs
subband(::ShearletSystem, ::AbstractArray, ::Integer)
similar_coefficients(::ShearletSystem, ::AbstractArray)
```

## From MultiscaleTransformsBase

The generic types and functions of the transform interface, and its
conformance tests; Shearlets re-exports the exported names.

```@docs
MultiscaleTransformsBase.MultiscaleTransformsBase
MultiscaleTransformsBase.AbstractMultiscaleTransform
MultiscaleTransformsBase.AbstractTransformWorkspace
MultiscaleTransformsBase.transform
MultiscaleTransformsBase.nsubbands
MultiscaleTransformsBase.SubbandVector
MultiscaleTransformsBase.SubbandLayout
MultiscaleTransformsBase.DefaultWorkspace
MultiscaleTransformsBase.check_workspace
MultiscaleTransformsBase.Testing
MultiscaleTransformsBase.Testing.test_transform
```

## Re-exported from MultiscaleFilterBanks

```@docs
FilterPair
CDF97
```

## Other MultiscaleFilterBanks utilities

Not re-exported by Shearlets; qualify them with `MultiscaleFilterBanks.`.

```@docs
MultiscaleFilterBanks.upsample_filter
MultiscaleFilterBanks.ReadOnlyTaps
```
