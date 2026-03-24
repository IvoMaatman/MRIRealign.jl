```@meta
CurrentModule = MRIRealign
```

# MRIRealign.jl

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

Call [`realign!`](@ref), which overwrites `img` with the aligned volumes
and returns the motion parameters — a `(6, t)` matrix where each column
is `[rx, ry, rz, tx, ty, tz]` (rotations in radians, translations in
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

### Reference modes

```julia
# Robust consensus across all pairwise alignments (default, slowest)
params = realign!(img; ref_mode=:consensus)

# Align to the temporal mean (fast, may be blurred)
params = realign!(img; ref_mode=:mean)

# Align to a specific time frame (fast, quality depends on that frame)
params = realign!(img; ref_mode=1)
```

### Smoothing

For noisy data, applying Gaussian smoothing before estimation can
improve robustness.  The `fwhm` keyword accepts a 3-tuple of
full-width-at-half-maximum values in voxel units:

```julia
params = realign!(img; fwhm=(5.0, 5.0, 5.0))
```

!!! note
    The default `fwhm=nothing` (no smoothing) differs from SPM's default
    of approximately 5 mm.  For noisy data, setting `fwhm` explicitly is
    recommended.

!!! note
    This tutorial assumes that the NIfTI headers of all files are
    identical and replaces the raw data with interpolated data.  To
    update the NIfTI header instead (preserving the original voxel
    data), use `realign=false` and write the parameters into the header.
    See [the NIfTI.jl documentation](https://github.com/JuliaNeuroscience/NIfTI.jl)
    for details.


## API Reference

### Main interface

```@docs
MRIRealign.realign!
```

### Geometric utilities

```@docs
MRIRealign.create_rotation_matrix
MRIRealign.create_affine_matrix
MRIRealign.params_from_rigid_affine
```