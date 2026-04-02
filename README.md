# MRIRealign.jl


| **Documentation**         | **Build Status**                      | **Test Coverage**               |
|:------------------------- |:------------------------------------- |:------------------------------- |
| [![][docs-img]][docs-url] | [![][gh-actions-img]][gh-actions-url] | [![][codecov-img]][codecov-url] |


MRIRealign.jl performs rigid-body (6-DOF) motion correction for 4-D MRI
time-series data.  It estimates three rotation angles and three
translations per volume by minimizing the sum of squared intensity
differences, then reslices (resamples) the volumes to undo the estimated
motion.

The algorithm follows the seminal paper by
[Friston et al.](https://doi.org/10.1002/hbm.460030303) and was heavily
inspired by [SPM's](https://www.fil.ion.ucl.ac.uk/spm/) `spm_realign`
function.  Key differences from SPM include:

* **Speed** — a Gauss–Newton trust-region optimizer with exact analytic
  Jacobians of the rotation matrix converges in fewer iterations than
  SPM's re-estimation loop.
* **Consensus estimation** — all time frames are aligned pairwise and a
  robust weighted consensus is computed via
  [iteratively reweighted least squares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)
  with geodesic rotation distance on SO(3).  This is less sensitive to
  the image quality of any single reference frame and avoids
  difficulties in mapping to a blurred temporal mean.

## Quick Start

On Unix systems, Julia can be installed with
```bash
curl -fsSL https://install.julialang.org | sh
```

and on Windows systems with
```
winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore
```
More detailed installation instructions can be found
[here](https://julialang.org/install/).

Thereafter, you can start Julia from the command line with
```bash
julia
```

### Loading data

This tutorial assumes that you have a folder at the path
`/path_to_files/` with NIfTI files of the format `mask.nii` and
`somename_1.nii`, `somename_2.nii`, … .  MRIRealign.jl does not include
I/O functions, so you are free to load data from
[NIfTI](https://github.com/JuliaNeuroscience/NIfTI.jl),
[DICOM](https://github.com/JuliaHealth/DICOM.jl),
[MAT](https://github.com/JuliaIO/MAT.jl),
[HDF5](https://github.com/JuliaIO/HDF5.jl) files, etc.

Install the packages once:

```julia
using Pkg
Pkg.add("MRIRealign")
Pkg.add("NIfTI")
```

Then load them:

```julia
using MRIRealign
using NIfTI
```

Change to the data directory:

```julia
cd("/path_to_files/")
```

Optionally, load a mask and convert it to a `BitArray`:

```julia
mask = round.(Bool, niread("mask.nii"))
```

Create a sorted list of volume file names (natural numeric order):

```julia
files = filter(f -> isfile(f) && f != "mask.nii", readdir())
files = sort(files, by = file -> parse(Int, match(r"\d+", file).match))
```

Allocate a 4-D array and read all volumes into it:

```julia
img = Array{Float64}(undef, size(mask)..., length(files))

for t in eachindex(files)
    img[:,:,:,t] .= niread(files[t]).raw
end
```

### Estimating and applying motion correction

Call `realign!`, which overwrites `img` with the aligned volumes and
returns the motion parameters — a `(6, t)` matrix where each column is
`[rx, ry, rz, tx, ty, tz]` (rotations in radians, translations in
voxels):

```julia
params = realign!(img; mask=mask)
```

Write the aligned images back to NIfTI files:

```julia
for t in eachindex(files)
    ni = niread(files[t])
    ni.raw .= img[:,:,:,t]
    niwrite(files[t], ni)
end
```

### Estimate-only workflow

To estimate motion parameters without modifying the images:

```julia
params = realign!(img; mask=mask, realign=false)
```

The returned `params` can later be applied with the two-argument form:

```julia
realign!(img, params)
```

> **Note:** This tutorial assumes that the NIfTI headers of all files are
> identical and replaces the raw data with interpolated data.  To update
> the NIfTI header instead (preserving the original voxel data), use
> `realign=false` and write the parameters into the header.  See
> [the NIfTI.jl documentation](https://github.com/JuliaNeuroscience/NIfTI.jl)
> for details.


[docs-img]: https://img.shields.io/badge/docs-latest%20release-blue.svg
[docs-url]: https://magneticresonanceimaging.github.io/MRIRealign.jl/stable

[gh-actions-img]: https://github.com/MagneticResonanceImaging/MRIRealign.jl/workflows/CI/badge.svg
[gh-actions-url]: https://github.com/MagneticResonanceImaging/MRIRealign.jl/actions

[codecov-img]: https://codecov.io/gh/MagneticResonanceImaging/MRIRealign.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/MagneticResonanceImaging/MRIRealign.jl

