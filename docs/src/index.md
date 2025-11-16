```@meta
CurrentModule = MRIRealign
```

# MRIRealign.jl

This package was heavily inspired by [SPM's](https://www.fil.ion.ucl.ac.uk/spm/) `spm_realign` function and aligns a time series of 3D MRI images with similar contrast. It minimizes the squared difference between the images in a given mask. The principal advantage over `spm_realign` is speed. Additionally, we implemented a *consensus* estimation, which aligns all time frames pairwise and calculates a consensus between all estimates. Compared to a single time frame as a reference, the consensus approach is less sensitive to the image quality of the reference frame. Compared with the *mean* time frame as a reference, it avoids difficulties in mapping to blurred images.


## Main Interface

```@docs
MRIRealign.realign!
```

## Helper functions

```@docs
MRIRealign.create_rotation_matrix
MRIRealign.create_affine_matrix
```