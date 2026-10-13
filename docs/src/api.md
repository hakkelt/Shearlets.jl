# API Reference

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
shearlet_layout
shearlet_nsubbands
shearlet_spectra
```

## Transforms

```@docs
shearlet_forward
shearlet_forward!
shearlet_inverse
shearlet_inverse!
ShearletWorkspace
make_shearlet_workspace
```

## Re-exported from MultiscaleFilterBanks

```@docs
FilterPair
CDF97
```

## Other MultiscaleFilterBanks utilities

Not re-exported by Shearlets; call it as `MultiscaleFilterBanks.upsample_filter`.

```@docs
MultiscaleFilterBanks.upsample_filter
```
