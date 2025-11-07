module ImageRealign

using Images
using Optim
using StaticArrays
using LinearAlgebra
using Interpolations
using Interpolations: gradient, hessian
using ImageFiltering
using Statistics

export estimate_motion_parameters


# --- Top-level functions ---
function estimate_motion_parameters(img::Array{<:Real,4}; ref_mode=:first, subsample=4, mask=nothing, fwhm=3.0)
    sx, sy, sz, nt = size(img)
    reference = ref_mode == :mean ? mean(img, dims=4)[:, :, :, 1] : img[:, :, :, 1]
    reference = smooth_image(reference, fwhm)

    center = voxelcenter(reference)
    motion_params = zeros(nt, 6)

    mask_inds = findall(mask)
    mask_inds = mask_inds[rand(1:subsample, length(mask_inds)).==1]

    Threads.@threads for t in 1:nt
        println("Estimating motion parameters $t / $nt ...")
        moving = img[:, :, :, t]
        moving = smooth_image(moving, fwhm)
        p_est = estimate_rigid(reference, moving, center, mask_inds)
        motion_params[t, :] .= p_est
    end

    return motion_params
end


# TODO
# function realign_volumes(img::Array{<:Real,4}; ref_mode=:first, subsample=4, mask=nothing, σ=3.0)
#     sx, sy, sz, nt = size(img)
#     center = voxelcenter(img[:,:,:,1])
#     aligned = Array{Float64,4}(undef, sx, sy, sz, nt)

#     Threads.@threads for t in 1:nt
#         println("Realigning volume $t / $nt ...")
#         A = rigid_affine_from_params(motion_params[t, :], center)
#         aligned[:,:,:,t] = warp_volume_itp(moving, A, (sx,sy,sz))
#     end

#     return motion_params
# end


## --- Rigid estimation ---
function estimate_rigid(reference, moving, center, mask_inds)
    fgh! = make_fgh_function(reference, moving, center, mask_inds)
    res = optimize(Optim.only_fgh!(fgh!), zeros(6), NewtonTrustRegion())
    return Optim.minimizer(res)
end


# --- Combined f, g, h! for Optim.only_fg! or Newton trust-region ---
function make_fgh_function(reference::Array{Float64,3}, moving::Array{Float64,3}, center, mask_inds)
    img_shape = size(moving)

    ref_itp = extrapolate(interpolate(reference, BSpline(Cubic())), 0.0)
    mov_itp = extrapolate(interpolate(moving, BSpline(Cubic())), 0.0)

    warped = zeros(Float64, img_shape)
    grad_field = [zeros(SVector{3,Float64}) for _ ∈ CartesianIndices(img_shape)]
    hess_field = [zeros(SMatrix{3,3,Float64}) for _ ∈ CartesianIndices(img_shape)]
    diff_vals = zeros(Float64, length(mask_inds))

    warp_grad_hess!(nothing, grad_field, hess_field, ref_itp, nothing, mask_inds)

    function fgh!(F, G, H, p)
        A = rigid_affine_from_params(p, center)
        warp_grad_hess!(warped, nothing, nothing, mov_itp, A, mask_inds)

        # Residuals
        @inbounds for (n, I) in enumerate(mask_inds)
            diff_vals[n] = reference[I] - warped[I]
        end

        # Objective value
        F = sum(abs2, diff_vals)

        # Initialize gradient and Hessian
        if G !== nothing; fill!(G, 0); end
        if H !== nothing; fill!(H, 0); end

        # Per-voxel contributions
        @inbounds for (n, I) in enumerate(mask_inds)
            x = I[1] - center[1]
            y = I[2] - center[2]
            z = I[3] - center[3]

            # ∂x/∂params (3×6 Jacobian)
            Jx = @SMatrix [
                0 z -y 1 0 0;
                -z 0 x 0 1 0;
                y -x 0 0 0 1
            ]

            gI = grad_field[I]      # ∂I/∂x, shape (3,)
            HI = hess_field[I]      # ∂²I/∂x², shape (3×3)
            r = diff_vals[n]

            if G !== nothing
                G .+= (-2r) .* (Jx' * gI)
            end

            if H !== nothing
                # Hessian term: 2*(JᵀJ - r * second)
                H .+= 2 .* (Jx' * (gI * gI') * Jx - r .* (Jx' * HI * Jx))
                # H .+= 2 .* (Jx' * (gI * gI') * Jx)   # Gauss-Newton approx
            end
        end
        return F
    end

    return fgh!
end


# --- warp, gradient, and hessian ---
function warp_grad_hess!(warped, grad_field, hess_field, itp, A, inds)
    for I in inds
        if A === nothing
            x, y, z = I[1], I[2], I[3]
        else
            v = A * SVector{4,Float64}(I[1], I[2], I[3], 1.0)
            x, y, z = v[1], v[2], v[3]
        end
        warped === nothing || (warped[I] = itp(x, y, z))
        grad_field === nothing || (grad_field[I] = gradient(itp, x, y, z)) # TODO: test gradient!
        hess_field === nothing || (hess_field[I] = hessian(itp, x, y, z))
    end
end

## --- utilities ---
"""
    smooth_image(img, fwhm; voxel_size=(1,1,1))

Smooth a 3D volume with a Gaussian kernel of given `fwhm` (in the same units as `voxel_size`).
Returns a Float64 array.
"""
function smooth_image(img::AbstractArray{<:Real,3}, fwhm::Real; voxel_size=(4.0, 4.0, 4.0))
    # Convert FWHM [mm] → σ [voxels]
    σ = ntuple(i -> (fwhm / (2√(2log(2)))) / voxel_size[i], 3)
    return imfilter(img, Kernel.gaussian(σ))
end


function rigid_affine_from_params(p, center)
    rx, ry, rz, tx, ty, tz = p
    Rx = @SMatrix [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)]
    Ry = @SMatrix [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)]
    Rz = @SMatrix [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1]
    R = Rz * Ry * Rx

    T_center_to_origin = Matrix{Float64}(I, 4, 4)
    T_center_to_origin[1:3, 4] .= -collect(center)

    T_back = Matrix{Float64}(I, 4, 4)
    T_back[1:3, 4] .= collect(center)

    Arot = Matrix{Float64}(I, 4, 4)
    Arot[1:3, 1:3] .= R

    Atrans = Matrix{Float64}(I, 4, 4)
    Atrans[1, 4] = tx
    Atrans[2, 4] = ty
    Atrans[3, 4] = tz

    return SMatrix{4,4,Float64}(Atrans * T_back * Arot * T_center_to_origin)
end

voxelcenter(vol) = ((size(vol, 1) + 1) / 2, (size(vol, 2) + 1) / 2, (size(vol, 3) + 1) / 2)


end