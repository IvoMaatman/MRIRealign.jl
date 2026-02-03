# MRIRealign.jl


| **Documentation**         | **Build Status**                      | **Test Coverage**               |
|:------------------------- |:------------------------------------- |:------------------------------- |
| [![][docs-img]][docs-url] | [![][gh-actions-img]][gh-actions-url] | [![][codecov-img]][codecov-url] |


This package aligns a time series of 3D MRI images with similar contrast, following the seminal paper by [Friston et al.](https://doi.org/10.1002/hbm.460030303) It minimizes the squared difference between the images in a given mask. The package was heavily inspired by [SPM's](https://www.fil.ion.ucl.ac.uk/spm/) `spm_realign` function.  The principal advantage over `spm_realign` is speed. Additionally, we implemented a *consensus* estimation, which aligns all time frames pairwise and uses [iteratively reweighted least squares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares) to calculate a consensus between all estimates. Compared to a single reference time frame, the consensus approach is less sensitive to the image quality of the reference frame. Compared with the *mean* time frame as a reference, it avoids difficulties in mapping to a blurred reference.

## Quick Tutorial

On Unix systems, Julia can be installed with
```Bash
curl -fsSL https://install.julialang.org | sh
```

and on Windows systems with
```
winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore
```
More detailed installation instructions can be found [here](https://julialang.org/install/).

Thereafter, you can start Julia from the command line with
```Bash
julia
```

This section assumes that you have a folder at the path `/path_to_files/` with NIfTI files of the format `mask.nii` and `somename_1.nii`, `somename_2.nii`, ... . Our package does not include loading functions, allowing users to load data from [NIfTI](https://github.com/JuliaNeuroscience/NIfTI.jl), [DICOM](https://github.com/JuliaHealth/DICOM.jl), [Matlab](https://github.com/JuliaIO/MAT.jl), [HDF5](https://github.com/JuliaIO/HDF5.jl) files etc.

The first time, the packages need to be installed with the package manager:

```@Julia
using Pkg
Pkg.add("MRIRealign")
Pkg.add("NIfTI")
```

Thereafter, we can use them:

```@Julia
using MRIRealign
using NIfTI
```

We can change the directory
```@Julia
cd("/path_to_files/")
```

and, optionally, load a mask and convert it to a binary mask
```@Julia
mask = round.(Bool, niread("mask.nii"))
```
Note that the `.` after round indicates a point-wise operation.

We can create a list of file names in the current folder, except for `mask.nii`, and sort them in natural order, i.e., 1, 2, 3, ... instead of the ASCII order 1, 10, 100, 101, ... .

```@Julia
files = filter(f -> isfile(f) && f != "mask.nii", readdir())
files = sort(files, by = file -> parse(Int, match(r"\d+", file).match))
```

Using the size of the mask, where `size(mask)...` returns the three dimensions separately, we can allocate an array and load all time frames into it:

```@Julia
img = Array{Float64}(undef, size(mask)..., length(files))

for t in eachindex(files)
    img[:,:,:,t] .= niread(files[t]).raw
end
```

Now we are all set to call the `realign!` function, which will overwrite `img` with the aligned volumes and return the motion parameters, i.e., 3 rotation and 3 translation parameters in this order:
```@Julia
params = realign!(img; mask=mask)
```

We can write the aligned images back to the NIfTI files:
```@Julia
for t in eachindex(files)
    ni = niread(files[t])
    ni.raw .= img[:,:,:,t]
    niwrite(files[t], ni)
end
```

Note that this tutorial assumes that the headers of all NIfTI files are identical and replaces the raw data with interpolated data. For changing the NIfTI header instead, we can call `params = realign!(img; mask=mask, realign=false)` and write `params` to the NIfTI header. For more information, refer to [the NIfTI.jl documentation](https://github.com/JuliaNeuroscience/NIfTI.jl).


[docs-img]: https://img.shields.io/badge/docs-latest%20release-blue.svg
[docs-url]: https://magneticresonanceimaging.github.io/MRIRealign.jl/stable

[gh-actions-img]: https://github.com/MagneticResonanceImaging/MRIRealign.jl/workflows/CI/badge.svg
[gh-actions-url]: https://github.com/MagneticResonanceImaging/MRIRealign.jl/actions

[codecov-img]: https://codecov.io/gh/MagneticResonanceImaging/MRIRealign.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/MagneticResonanceImaging/MRIRealign.jl

