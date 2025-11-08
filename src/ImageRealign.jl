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
function estimate_motion_parameters(img::Array{T,4}; ref_mode=:first, subsample=4, mask=nothing, fwhm=T(3)) where {T<:Real}
    img_s = (fwhm === nothing || fwhm == 0) ? img : smooth_image(img, fwhm)

    reference = ref_mode == :mean ? mean(img_s, dims=4)[:, :, :, 1] : img_s[:, :, :, 1]

    center = voxelcenter(reference)
    motion_params = zeros(size(img_s, 4), 6)

    _mask_inds = findall(mask)
    mask_inds = _mask_inds[rand(1:subsample, length(_mask_inds)).==1]

    Threads.@threads for t ∈ axes(img_s, 4)
        moving = img_s[:, :, :, t]
        fgh! = make_fgh_function(reference, moving, center, mask_inds)
        res = optimize(Optim.only_fgh!(fgh!), zeros(6), NewtonTrustRegion())
        motion_params[t, :] .= Optim.minimizer(res)
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


# --- Combined f, g, h! for Optim.only_fg! or Newton trust-region ---
function make_fgh_function(reference::AbstractArray{T,3}, moving::AbstractArray{T,3}, center, mask_inds) where T
    ref_itp = extrapolate(interpolate(reference, BSpline(Cubic())), 0.0)
    mov_itp = extrapolate(interpolate(moving, BSpline(Cubic())), 0.0)

    grad_field = [gradient(ref_itp, i[1], i[2], i[3]) for i ∈ mask_inds]
    hess_field = [ hessian(ref_itp, i[1], i[2], i[3]) for i ∈ mask_inds]

    diff_vals = similar(mask_inds, T)

    function fgh!(F, G, H, p)
        # Residuals
        A = rigid_affine_from_params(p, center)
        @inbounds for (n, i) in enumerate(mask_inds)
            v = A * SVector{4,T}(i[1], i[2], i[3], 1)
            diff_vals[n] = reference[i] - mov_itp(v[1], v[2], v[3])
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
            HI = hess_field[i]      # ∂²I/∂x², shape (3×3)
            r = diff_vals[i]

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


## --- utilities ---
"""
    smooth_image(img, fwhm; voxel_size=(1,1,1))

Smooth a 3D volume with a Gaussian kernel of given `fwhm` (in the same units as `voxel_size`).
Returns a Float64 array.
"""
function smooth_image(img::AbstractArray{T,3}, fwhm::T; voxel_size=ntuple(_ -> T(4), 3)) where {T<:Real}
    # Convert FWHM [mm] → σ [voxels]
    σ = ntuple(i -> (fwhm / (2√(2log(2)))) / voxel_size[i], 3)
    img = imfilter(img, Kernel.gaussian(σ))
    return img
end

function smooth_image(img::AbstractArray{T,4}, fwhm::T; voxel_size=ntuple(_ -> T(4), 3)) where {T<:Real}
    img_s = similar(img)
    Threads.@threads for it ∈ axes(img, 4)
        @views img_s[:, :, :, it] .= smooth_image(img[:, :, :, it], fwhm; voxel_size)
    end
    return img_s
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