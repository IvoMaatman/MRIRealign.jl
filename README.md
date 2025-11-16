# MRIRealign.jl


| **Documentation**         | **Build Status**                      |
|:------------------------- |:------------------------------------- |
| [![][docs-img]][docs-url] | [![][gh-actions-img]][gh-actions-url] |
|                           | [![][codecov-img]][codecov-url]       |


This package was heavily inspired by [SPM's](https://www.fil.ion.ucl.ac.uk/spm/) `spm_realign` function and aligns a time series of 3D MRI images with similar contrast. It minimizes the squared difference between the images in a given mask. The principal advantage over `spm_realign` is speed. Additionally, we implemented a *consensus* estimation, which aligns all time frames pairwise and calculates a consensus between all estimates. Compared to a single time frame as a reference, the consensus approach is less sensitive to the image quality of the reference frame. Compared with the *mean* time frame as a reference, it avoids difficulties in mapping to blurred images.

[docs-img]: https://img.shields.io/badge/docs-latest%20release-blue.svg
[docs-url]: https://magneticresonanceimaging.github.io/MRIRealign.jl/dev/

[gh-actions-img]: https://github.com/MagneticResonanceImaging/MRIRealign.jl/workflows/CI/badge.svg
[gh-actions-url]: https://github.com/MagneticResonanceImaging/MRIRealign.jl/actions

[codecov-img]: https://codecov.io/gh/MagneticResonanceImaging/MRIRealign.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/MagneticResonanceImaging/MRIRealign.jl

