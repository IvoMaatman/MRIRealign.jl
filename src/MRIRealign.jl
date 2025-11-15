module MRIRealign

using Images
using Optim
using StaticArrays
using LinearAlgebra
using Interpolations
using Interpolations: gradient, hessian
using ImageFiltering
using Statistics
using OhMyThreads

export realign!, create_rotation_matrix, create_affine_matrix


# --- Top-level functions ---
"""
    realign!(img; center=size(img)[1:3] .÷ 2, ref_mode=:consensus, mask=trues(size(img)[1:3]), fwhm=nothing, realign=true)
    realign!(img, motion_params; center=size(img)[1:3] .÷ 2)

Estimate motion parameters and/or realign the images.

# Required argument
    - `img::AbstractArray{T,4}`: Array of images with the dimensions `x, y, z, t`, i.e., 3D images with time in the 4th dimension. The type T can be real or complex valued. If complex-valued, the motion estimation will be performed on the absolute value; otherwise, on img, i.e., allowing for negative values.

# Keyword arguments if motion parameters are unknown
    - `center=size(img)[1:3] .÷ 2`: Point around which the images are rotated (relevant only for the estimated motion parameters, not the alignment).
    - `ref_mode: `:consensus` (default), `:mean`, or an integer. `:consensus` estimates motion parameters pairwise for all timeframes and computes a consensus, which is helpful if any single reference might have poor image quality. This comes at the cost of a `t`-fold increase in computation time. `:mean` estimates the motion parameters wrt. to the mean of all images. This is fast, but might have inferior precision if the mean is blurred by substantial motion. Providing an integer aligns the images wrt. to the `ref_mode`th time frame, which is fast, but works only reliably if this time frame has good image quality.
    - `mask=trues(size(img)[1:3])`: bitmask at which the frames are compared.
    - `fwhm=nothing`: 3-Tuple of the full width at half maximum values of an optional Gaussian smoothing kernel along each dimension, in units of voxels. The default setting (`nothing`) is to perform no smoothing.
    - `realign=true`: If true, the argument `img` will be overwritten inline with the aligned images. If `false`, this function estimates the motion parameters, but does not align the images.

# Optional arguments if the motion parameters are already known
    - `motion_params::AbstractMatrix`: If the motion parameters are known, e.g., by a previous run of this function, they can be provided to skip the estimation step. The dimensions of this matrix are `6 × T`, capturing in the first dimension the motion parameters, in the order `rx, ry, rz, tx, ty, tz`, with the rotations `r` and the translations `t`. T is the number of time frames.
    - `center=size(img)[1:3] .÷ 2)`: Point around which `motion_params` are applied.

With the appropriate settings (see above), the aligned timeframes are written inline into `img`. The function always returns the estimated motion parameters, where all rotations are in radians and translations in voxels.
"""
function realign!(img::AbstractArray{Tin,4};
    center=size(img)[1:3] .÷ 2,
    ref_mode=:consensus,
    mask=trues(size(img)[1:3]),
    fwhm=nothing::Union{Nothing,NTuple{3}},
    realign=true
) where Tin

    if Tin <: Complex
        _img = abs.(img)
        Treal = real(Tin)
    else
        _img = img
        Treal = Tin
    end
    T = Float64 # lower precision results in gradient inacurracies

    if ref_mode == :consensus
        t_refs = axes(_img, 4)
    elseif ref_mode == :mean
        _img = cat(_img, mean(_img, dims=4); dims=4)
        t_refs = size(_img, 4)
    elseif typeof(ref_mode) <: Integer
        t_refs = ref_mode
    else
        error("ref_mode must either be `:consensus`, `:mean`, or an integer")
    end

    img_s = (fwhm === nothing || all(fwhm .== 0)) ? _img : smooth_image(_img, fwhm)

    # interpolate all time frames
    @views Tint = typeof(_interpolate(img_s[:, :, :, 1]))
    img_itp = Vector{Tint}(undef, size(img_s, 4))
    @tasks for t ∈ eachindex(img_itp)
        vol = img_s[:, :, :, t]
        vol ./= quantile(vec(vol), T(0.9))
        img_itp[t] = _interpolate(vol)
    end

    # random shifts seem to help with the speed of convertion (cf. SPM)
    mask_inds = [Tuple(idx) .+ rand(NTuple{3,T}) .- T(0.5) for idx ∈ findall(mask)]

    _motion_params = Array{T}(undef, 6, length(img_itp), length(t_refs))
    for (i_ref, t_ref) ∈ enumerate(t_refs)
        reference = [img_itp[t_ref](idx[1], idx[2], idx[3]) for idx in mask_inds]
        grad_field = [gradient(img_itp[t_ref], idx[1], idx[2], idx[3]) for idx ∈ mask_inds]
        # hess_field = [hessian(img_itp[t_ref], idx[1], idx[2], idx[3]) for idx ∈ mask_inds]
        hess_field = nothing # using the Gauss-Newton approximation

        p0 = zeros(T, 6)
        @tasks for t ∈ axes(img_s, 4)
            @local diff_vals = similar(mask_inds, T)

            fgh! = make_fgh_function(reference, img_itp[t], center, mask_inds, grad_field, hess_field, diff_vals)
            res = optimize(Optim.only_fgh!(fgh!), p0, NewtonTrustRegion())
            _motion_params[:, t, i_ref] .= Optim.minimizer(res)
        end
    end

    motion_params = if ref_mode == :consensus
        weighted_mean_estimate!(_motion_params; r=mean(size(img)[1:3]))
    elseif ref_mode == :mean
        _motion_params[:, 1:end-1, 1]
    else
        dropdims(_motion_params, dims=3)
    end

    if realign
        realign!(img, motion_params; center)
    end

    return Treal.(motion_params)
end


function realign!(img::AbstractArray{T,4}, motion_params; center=size(img)[1:3] .÷ 2) where T
    @tasks for t ∈ axes(img, 4)
        vol = @view img[:, :, :, t]
        img_itp = _interpolate(vol)
        A = create_affine_matrix(motion_params[:, t], center)
        @inbounds for idx ∈ CartesianIndices(vol)
            v = A * SVector{4,Float64}(idx[1], idx[2], idx[3], 1)
            vol[idx] = img_itp(v[1], v[2], v[3])
        end
    end
    return motion_params
end


# --- Combined f, g, h! for Optim.only_fg! or Newton trust-region ---
function make_fgh_function(reference::AbstractVector{T}, mov_itp, center, mask_inds, grad_field, hess_field, diff_vals) where T
    function fgh!(F, G, H, p)
        # Residuals
        A = create_affine_matrix(p, center)
        @inbounds for (n, i) ∈ enumerate(mask_inds)
            v = A * SVector{4,T}(i[1], i[2], i[3], 1)
            diff_vals[n] = reference[n] - mov_itp(v[1], v[2], v[3])
        end

        # Objective value
        F = sum(abs2, diff_vals)

        # Initialize gradient and Hessian
        G === nothing || fill!(G, 0)
        H === nothing || fill!(H, 0)

        # Per-voxel contributions
        @inbounds for i ∈ eachindex(mask_inds)
            x = mask_inds[i][1] - center[1]
            y = mask_inds[i][2] - center[2]
            z = mask_inds[i][3] - center[3]

            # ∂x/∂params (3×6 Jacobian)
            Jx = @SMatrix [
                0 z -y 1 0 0;
                -z 0 x 0 1 0;
                y -x 0 0 0 1
            ]

            gI = grad_field[i]      # ∂I/∂x, shape (3,)
            r = diff_vals[i]

            if G !== nothing
                G .+= (-2r) .* (Jx' * gI)
            end

            if H !== nothing
                if hess_field === nothing
                    H .+= 2 .* (Jx' * (gI * gI') * Jx)   # Gauss-Newton approx
                else
                    # Hessian term: 2*(JᵀJ - r * second)
                    HI = hess_field[i]      # ∂²I/∂x², shape (3×3)
                    H .+= 2 .* (Jx' * (gI * gI') * Jx - r .* (Jx' * HI * Jx))
                end
            end
        end
        return F
    end

    return fgh!
end


## --- utilities ---
"""
    smooth_image(img, fwhm)

Smooth a 3D volume with a Gaussian kernel of given `fwhm` is a three-tuple (in units of voxel).
Returns an array.
"""
function smooth_image(img::AbstractArray{T,3}, fwhm::NTuple{3}) where {T<:Real}
    σ = T.(fwhm ./ (2√(2log(2))))
    img = imfilter(img, Kernel.gaussian(σ))
    return img
end

function smooth_image(img::AbstractArray{T,4}, fwhm::NTuple{3}) where {T<:Real}
    img_s = similar(img)
    @tasks for it ∈ axes(img, 4)
        @views img_s[:, :, :, it] .= smooth_image(img[:, :, :, it], fwhm)
    end
    return img_s
end

_interpolate(x) = extrapolate(interpolate(x, BSpline(Cubic())), Interpolations.Flat())


"""
    create_rotation_matrix(rx, ry, rz)
    create_rotation_matrix(p) = create_rotation_matrix(p[1], p[2], p[3])

Calculate the rotation matrix for three rotations `rx, ry, rz`, in radians. In this package, we use the convention `R = Rz * Ry * Rx`.
"""
create_rotation_matrix(p) = create_rotation_matrix(p[1], p[2], p[3])

function create_rotation_matrix(rx, ry, rz)
    Rx = @SMatrix [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)]
    Ry = @SMatrix [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)]
    Rz = @SMatrix [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1]
    R = Rz * Ry * Rx
    return R
end

"""
    create_affine_matrix(p, center)

Calculate the affine matrix from `p = rx, ry, rz, tx, ty, tz`. The rotations `r` are in radians and the translations `t` in voxels. The argument `center` takes a 3-Tuple (or vector of length 3) with the center around which the images are rotated.
"""
function create_affine_matrix(p, center)
    rx, ry, rz, tx, ty, tz = p

    Rx = @SMatrix [1 0 0 0; 0 cos(rx) -sin(rx) 0; 0 sin(rx) cos(rx) 0; 0 0 0 1]
    Ry = @SMatrix [cos(ry) 0 sin(ry) 0; 0 1 0 0; -sin(ry) 0 cos(ry) 0; 0 0 0 1]
    Rz = @SMatrix [cos(rz) -sin(rz) 0 0; sin(rz) cos(rz) 0 0; 0 0 1 0; 0 0 0 1]
    R = Rz * Ry * Rx

    T_center_to_origin = @SMatrix [
        1 0 0 -center[1]
        0 1 0 -center[2]
        0 0 1 -center[3]
        0 0 0 1
    ]

    T_back = @SMatrix [
        1 0 0 center[1]
        0 1 0 center[2]
        0 0 1 center[3]
        0 0 0 1
    ]

    Atrans = @SMatrix [
        1 0 0 tx
        0 1 0 ty
        0 0 1 tz
        0 0 0 1
    ]

    return Atrans * T_back * R * T_center_to_origin
end

function params_from_rigid_affine(A, center)
    T = eltype(A)
    R = @SMatrix [A[1, 1] A[1, 2] A[1, 3];
        A[2, 1] A[2, 2] A[2, 3];
        A[3, 1] A[3, 2] A[3, 3]]
    t_total = @SVector [A[1, 4], A[2, 4], A[3, 4]]

    # Extract Euler angles for Rz * Ry * Rx
    ry = asin(-R[3, 1])
    if abs(R[3, 1]) < 1 - T(1e-8)
        rx = atan(R[3, 2], R[3, 3])
        rz = atan(R[2, 1], R[1, 1])
    else # Gimbal lock case
        rx = zero(T)
        if R[3, 1] < 0
            rz = atan(-R[1, 2], -R[1, 3])
        else
            rz = atan(R[1, 2], R[1, 3])
        end
    end

    # Compute explicit translation parameters
    t = t_total - ((I(3) - R) * SVector(center))

    return @SVector [rx, ry, rz, t[1], t[2], t[3]]
end

function weighted_mean_estimate!(estimates; r=1, max_iter=100, tol=1e-6)
    # bring all estimates in the same frame of reference (iframe = end÷2)
    for iref ∈ axes(estimates, 3)
        @views A0 = create_affine_matrix(estimates[:, end÷2, iref], (0, 0, 0))
        for iframe ∈ axes(estimates, 2)
            @views A = create_affine_matrix(estimates[:, iframe, iref], (0, 0, 0))
            estimates[:, iframe, iref] = params_from_rigid_affine(A / A0, (0, 0, 0))
        end
    end

    consensus = mean(estimates; dims=3) # Initial guess

    for _ = 1:max_iter
        consensus_old = consensus

        residuals = estimates .- consensus
        residuals[1:3, :, :] .*= r # weight rotations by the radius
        weights = 1 ./ (1 .+ sqrt.(sum(abs2, residuals; dims=1:2))) # dims=1:2 calculates the weights for all 6 parameters jointly, while dims=2 calculates them for each parameter separately
        consensus = sum((weights .* estimates); dims=3) ./ sum(weights; dims=3) # Compute new weighted mean

        if norm(consensus .- consensus_old) < tol
            break
        end
    end
    consensus = dropdims(consensus, dims=3)
    return consensus
end

end