module ImageRealign

using Images
using Optim
using StaticArrays
using LinearAlgebra
using Interpolations
using Interpolations: gradient, hessian
using ImageFiltering
using Statistics
using OhMyThreads

export estimate_motion_parameters


# --- Top-level functions ---
function estimate_motion_parameters(img::Array{T,4}; ref_mode=:consensus, subsample=4, mask=nothing, fwhm=T(3)) where {T<:Real}
    img_s = (fwhm === nothing || fwhm == 0) ? img : smooth_image(img, fwhm)

    _interpolate(x) = extrapolate(interpolate(x, BSpline(Cubic())), Interpolations.Flat())
    @views Tint = typeof(_interpolate(img_s[:, :, :, 1]))
    img_itp = Vector{Tint}(undef, size(img_s, 4))
    @tasks for t ∈ eachindex(img_itp)
        @views img_itp[t] = _interpolate(img_s[:, :, :, t])
    end

    @views center = voxelcenter(img_s[:, :, :, 1])

    t_refs = ref_mode == :consensus ? (1:length(img_itp)) : ref_mode

    # random shifts seem to help with the speed of convertion (cf. SPM)
    mask_inds = [Tuple(idx) .+ rand(NTuple{3,T}) .- T(0.5) for idx ∈ findall(mask)]

    # _mask_inds = Tuple.(findall(mask))
    # _mask_inds = [_mask_inds; [idx .+ T.((0.5, 0, 0)) for idx ∈ _mask_inds]]
    # _mask_inds = [_mask_inds; [idx .+ T.((0, 0.5, 0)) for idx ∈ _mask_inds]]
    # _mask_inds = [_mask_inds; [idx .+ T.((0, 0, 0.5)) for idx ∈ _mask_inds]]
    # mask_inds = [idx .+ rand(NTuple{3,T}) ./ 2 .- T(0.25) for idx ∈ _mask_inds]

    motion_params = zeros(6, length(img_itp), length(t_refs))
    for (i_ref, t_ref) ∈ enumerate(t_refs)
        reference = [img_itp[t_ref](idx[1], idx[2], idx[3]) for idx in mask_inds]
        grad_field = [gradient(img_itp[t_ref], idx[1], idx[2], idx[3]) for idx ∈ mask_inds]
        hess_field = [hessian(img_itp[t_ref], idx[1], idx[2], idx[3]) for idx ∈ mask_inds]

        p0 = zeros(T, 6)
        @tasks for t ∈ axes(img_s, 4)
            @local diff_vals = similar(mask_inds, T)

            fgh! = make_fgh_function(reference, img_itp[t], center, mask_inds, grad_field, hess_field, diff_vals)
            res = optimize(Optim.only_fgh!(fgh!), p0, NewtonTrustRegion())
            motion_params[:, t, i_ref] .= Optim.minimizer(res)
        end
    end

    if ref_mode == :consensus
        return weighted_mean_estimate(motion_params)
    else
        return motion_params
    end
end


# TODO
# function realign_volumes(img::Array{<:Real,4}; ref_mode=:first, subsample=4, mask=nothing, σ=3.0)
#     sx, sy, sz, nt = size(img)
#     center = voxelcenter(img[:,:,:,1])
#     aligned = Array{Float64,4}(undef, sx, sy, sz, nt)

#     @tasks for t in 1:nt
#         println("Realigning volume $t / $nt ...")
#         A = rigid_affine_from_params(motion_params[t, :], center)
#         aligned[:,:,:,t] = warp_volume_itp(moving, A, (sx,sy,sz))
#     end

#     return motion_params
# end


# --- Combined f, g, h! for Optim.only_fg! or Newton trust-region ---
function make_fgh_function(reference::AbstractVector{T}, mov_itp, center, mask_inds, grad_field, hess_field, diff_vals) where T
    function fgh!(F, G, H, p)
        # Residuals
        A = rigid_affine_from_params(p, center)
        @inbounds for (n, i) in enumerate(mask_inds)
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
Returns an array.
"""
function smooth_image(img::AbstractArray{T,3}, fwhm::T; voxel_size=ntuple(_ -> T(4), 3)) where {T<:Real}
    # Convert FWHM [mm] → σ [voxels]
    σ = ntuple(i -> (fwhm / (2√(2log(2)))) / voxel_size[i], 3)
    img = imfilter(img, Kernel.gaussian(σ))
    return img
end

function smooth_image(img::AbstractArray{T,4}, fwhm::T; voxel_size=ntuple(_ -> T(4), 3)) where {T<:Real}
    img_s = similar(img)
    @tasks for it ∈ axes(img, 4)
        @views img_s[:, :, :, it] .= smooth_image(img[:, :, :, it], fwhm; voxel_size)
    end
    return img_s
end


function rigid_affine_from_params(p, center)
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

voxelcenter(vol) = ((size(vol, 1) + 1) / 2, (size(vol, 2) + 1) / 2, (size(vol, 3) + 1) / 2)


function weighted_mean_estimate(estimates; max_iter=100, tol=1e-6)
    consensus = mean(estimates; dims=3) # Initial guess

    for _ = 1:max_iter
        residuals = estimates .- consensus
        weights = 1 ./ (1 .+ sqrt.(sum(abs2, residuals; dims=1:2))) #! dims=1:2 calculates the weights for all 6 parameters jointly, while dims=2 calculates them for each parameter separately

        consensus_old = consensus
        consensus = sum((weights .* estimates); dims=3) ./ sum(weights; dims=3) # Compute new weighted mean

        if norm(consensus .- consensus_old) < tol
            break
        end
    end
    consensus = dropdims(consensus, dims=3)
    return consensus
end

end