module MRIRealign

using Images
using NLSolversBase
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

    # random shifts seem to help with the speed of convergence (cf. SPM)
    mask_inds = [SVector{3,T}(Tuple(idx)) + SVector{3,T}(rand(T), rand(T), rand(T)) - SVector{3,T}(0.5, 0.5, 0.5) for idx ∈ findall(mask)]

    # Precompute centered coordinates (mask position minus center) — constant across all references and optimizer iterations
    c_svec = SVector{3,T}(T(center[1]), T(center[2]), T(center[3]))
    xyz_centered = [ind - c_svec for ind ∈ mask_inds]

    _motion_params = Array{T}(undef, 6, length(img_itp), length(t_refs))
    for (i_ref, t_ref) ∈ enumerate(t_refs)
        reference = [img_itp[t_ref](ind[1], ind[2], ind[3]) for ind ∈ mask_inds]
        grad_field = [SVector{3,T}(gradient(img_itp[t_ref], ind[1], ind[2], ind[3])) for ind ∈ mask_inds]
        # hess_field = [hessian(img_itp[t_ref], idx[1], idx[2], idx[3]) for idx ∈ mask_inds]
        hess_field = nothing # using the Gauss-Newton approximation

        p0 = zeros(T, 6)
        @tasks for t ∈ axes(img_s, 4)
            @local diff_vals = similar(mask_inds, T)

            fgh! = make_fgh_function(reference, img_itp[t], center, mask_inds, xyz_centered, grad_field, hess_field, diff_vals)
            res = optimize(NLSolversBase.only_fgh!(fgh!), p0, NewtonTrustRegion())
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
function make_fgh_function(reference::AbstractVector{T}, mov_itp, center, mask_inds, xyz_centered, grad_field, hess_field, diff_vals) where T
    function fgh!(F, G, H, p)
        # Residuals
        A = create_affine_matrix(p, center)
        @inbounds for (n, ind) ∈ enumerate(mask_inds)
            v = A * SVector{4,T}(ind[1], ind[2], ind[3], 1)
            diff_vals[n] = reference[n] - mov_itp(v[1], v[2], v[3])
        end

        # Objective value
        F = sum(abs2, diff_vals)

        # Initialize gradient and Hessian
        G === nothing || fill!(G, 0)
        H === nothing || fill!(H, 0)

        # Precompute rotation matrix derivatives for exact Jacobian
        rx, ry, rz = p[1], p[2], p[3]
        sr, cr = sincos(rx)
        sp, cp = sincos(ry)
        sy, cy = sincos(rz)

        # ∂R/∂rx
        dRdrx = @SMatrix [
             0                     cy*sp*cr+sy*sr   -cy*sp*sr+sy*cr;
             0                     sy*sp*cr-cy*sr   -sy*sp*sr-cy*cr;
             0                     cp*cr            -cp*sr
        ]
        # ∂R/∂ry
        dRdry = @SMatrix [
            -cy*sp   cy*cp*sr   cy*cp*cr;
            -sy*sp   sy*cp*sr   sy*cp*cr;
            -cp     -sp*sr     -sp*cr
        ]
        # ∂R/∂rz
        dRdrz = @SMatrix [
            -sy*cp  -sy*sp*sr-cy*cr  -sy*sp*cr+cy*sr;
             cy*cp   cy*sp*sr-sy*cr   cy*sp*cr+sy*sr;
             0       0                0
        ]

        # Per-voxel contributions — direct dot-product formulation avoids
        # constructing the 3×6 Jacobian matrix Jx and the matrix-vector
        # product Jx' * gI.  Instead we compute JtgI (the 6-vector) directly:
        #   JtgI[1:3] = [dot(dRdrx*xyz, gI), dot(dRdry*xyz, gI), dot(dRdrz*xyz, gI)]
        #   JtgI[4:6] = gI                   (identity block for translations)
        @inbounds for i ∈ eachindex(mask_inds)
            xyz = xyz_centered[i]    # precomputed: mask_inds[i] - center
            gI  = grad_field[i]      # ∂I/∂x, shape (3,)
            r   = diff_vals[i]

            # Rotation part: dot(dR * xyz, gI) for each of rx, ry, rz
            drx_xyz = dRdrx * xyz
            dry_xyz = dRdry * xyz
            drz_xyz = dRdrz * xyz
            JtgI = SVector{6,T}(
                dot(drx_xyz, gI),
                dot(dry_xyz, gI),
                dot(drz_xyz, gI),
                gI[1], gI[2], gI[3]  # translation part (identity block)
            )

            if G !== nothing
                G .+= (-2r) .* JtgI
            end

            if H !== nothing
                H .+= 2 .* (JtgI * JtgI')   # Gauss-Newton approx (rank-1 update)
                if hess_field !== nothing
                    # Full Hessian correction requires the actual Jacobian matrix
                    Jx = hcat(drx_xyz, dry_xyz, drz_xyz,
                              SMatrix{3,3,T}(1,0,0, 0,1,0, 0,0,1))
                    HI = hess_field[i]       # ∂²I/∂x², shape (3×3)
                    H .-= (2r) .* (Jx' * HI * Jx)
                end
            end
        end
        return F
    end

    return fgh!
end


## --- utilities ---
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
    R = create_rotation_matrix(rx, ry, rz)
    c = SVector{3}(center[1], center[2], center[3])
    t = SVector{3}(tx, ty, tz) + c - R * c  # = translation + center - R * center
    @SMatrix [
        R[1,1] R[1,2] R[1,3] t[1];
        R[2,1] R[2,2] R[2,3] t[2];
        R[3,1] R[3,2] R[3,3] t[3];
        0      0      0      1
    ]
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

    nframes = size(estimates, 2)
    nrefs   = size(estimates, 3)
    consensus = Matrix{Float64}(undef, 6, nframes)

    for iframe ∈ 1:nframes
        # Convert all reference estimates for this frame to quaternions + translations
        quats = Vector{SVector{4,Float64}}(undef, nrefs)
        trans = Vector{SVector{3,Float64}}(undef, nrefs)
        for iref ∈ 1:nrefs
            p = @view estimates[:, iframe, iref]
            R = create_rotation_matrix(p[1], p[2], p[3])
            quats[iref] = _rotmat_to_quat(R)
            trans[iref] = SVector{3,Float64}(p[4], p[5], p[6])
        end

        # Initial consensus: unweighted mean
        q_mean, t_mean = _quat_weighted_mean(quats, trans, ones(nrefs))

        for _ = 1:max_iter
            q_old = q_mean
            t_old = t_mean

            # Compute weights based on geodesic rotation distance + translation distance
            w = Vector{Float64}(undef, nrefs)
            for iref ∈ 1:nrefs
                dot_val = clamp(abs(dot(quats[iref], q_mean)), 0.0, 1.0)
                rot_dist = 2 * acos(dot_val) * r  # geodesic distance scaled by radius
                trans_dist = norm(trans[iref] - t_mean)
                w[iref] = 1 / (1 + sqrt(rot_dist^2 + trans_dist^2))
            end

            q_mean, t_mean = _quat_weighted_mean(quats, trans, w)

            if norm(t_mean - t_old) + 2 * acos(clamp(abs(dot(q_mean, q_old)), 0.0, 1.0)) < tol
                break
            end
        end

        # Convert quaternion back to Euler angles + translation
        R_mean = _quat_to_rotmat(q_mean)
        A_mean = SMatrix{4,4,Float64}(
            R_mean[1,1], R_mean[2,1], R_mean[3,1], 0,
            R_mean[1,2], R_mean[2,2], R_mean[3,2], 0,
            R_mean[1,3], R_mean[2,3], R_mean[3,3], 0,
            t_mean[1],   t_mean[2],   t_mean[3],   1)
        consensus[:, iframe] .= params_from_rigid_affine(A_mean, (0, 0, 0))
    end
    return consensus
end

function _quat_weighted_mean(quats, trans, weights)
    # Weighted quaternion mean via dominant eigenvector of ∑ wᵢ qᵢ qᵢᵀ
    M = zeros(MMatrix{4,4,Float64})
    t_mean = zero(MVector{3,Float64})
    w_sum = zero(Float64)
    for i ∈ eachindex(quats)
        q = quats[i]
        M .+= weights[i] .* (q * q')
        t_mean .+= weights[i] .* trans[i]
        w_sum += weights[i]
    end
    t_mean ./= w_sum

    E = eigen(Symmetric(Matrix(M)))
    q_mean = SVector{4,Float64}(E.vectors[:, end])
    q_mean = q_mean[1] < 0 ? -q_mean : q_mean  # consistent sign
    return q_mean, SVector{3,Float64}(t_mean)
end

function _rotmat_to_quat(R)
    # Shepperd's method: convert 3×3 rotation matrix to unit quaternion [w, x, y, z]
    tr = R[1,1] + R[2,2] + R[3,3]
    if tr > 0
        s = 2 * sqrt(tr + 1)
        w = s / 4
        x = (R[3,2] - R[2,3]) / s
        y = (R[1,3] - R[3,1]) / s
        z = (R[2,1] - R[1,2]) / s
    elseif R[1,1] > R[2,2] && R[1,1] > R[3,3]
        s = 2 * sqrt(1 + R[1,1] - R[2,2] - R[3,3])
        w = (R[3,2] - R[2,3]) / s
        x = s / 4
        y = (R[1,2] + R[2,1]) / s
        z = (R[1,3] + R[3,1]) / s
    elseif R[2,2] > R[3,3]
        s = 2 * sqrt(1 + R[2,2] - R[1,1] - R[3,3])
        w = (R[1,3] - R[3,1]) / s
        x = (R[1,2] + R[2,1]) / s
        y = s / 4
        z = (R[2,3] + R[3,2]) / s
    else
        s = 2 * sqrt(1 + R[3,3] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = (R[1,3] + R[3,1]) / s
        y = (R[2,3] + R[3,2]) / s
        z = s / 4
    end
    q = SVector{4,Float64}(w, x, y, z)
    return q / norm(q)
end

function _quat_to_rotmat(q)
    # Convert unit quaternion [w, x, y, z] to 3×3 rotation matrix
    w, x, y, z = q
    @SMatrix [
        1-2(y^2+z^2)  2(x*y-z*w)    2(x*z+y*w);
        2(x*y+z*w)    1-2(x^2+z^2)  2(y*z-x*w);
        2(x*z-y*w)    2(y*z+x*w)    1-2(x^2+y^2)
    ]
end

end