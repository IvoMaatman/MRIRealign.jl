using ImageRealign
using Test
using ImagePhantoms
using Interpolations
using StaticArrays
using FiniteDifferences

## create phantom
shape = (64, 64, 64)
ob = ellipsoid(ellipsoid_parameters(fovs=(64,64,50)))
image = phantom(-shape[1]÷2+1:shape[1]÷2, -shape[2]÷2+1:shape[2]÷2, -shape[3]÷2+1:shape[3]÷2, ob, 3)
image = min.(image, 1.1)
image .-= 0.9
image = max.(image, 0)

center = (20, 10, 5)
width = (25, 35, 15)
angles = (π/6, 0, 0)
ob1 = gauss3(center, width, angles, 1.0f0)

center = (-15, 0, 5)
width = (5, 5, 15)
angles = (0, π/2, 0)
ob2 = gauss3(center, width, angles, 1.0f0)

center = (0, 0, 15)
width = (5, 5, 15)
angles = (0, 0, 0)
ob3 = gauss3(center, width, angles, 1.0f0)

image .+= 0.5 .* phantom(-shape[1]÷2+1:shape[1]÷2, -shape[2]÷2+1:shape[2]÷2, -shape[3]÷2+1:shape[3]÷2, [ob1, ob2, ob3], 3)
img_itp = extrapolate(interpolate(image, BSpline(Cubic())), Interpolations.Flat())


## test gradients
inds = [Tuple(idx) .+ rand(NTuple{3,Float64}) .- 0.5 for idx ∈ CartesianIndices(image)]
# inds = CartesianIndices(image)
reference = [img_itp(idx[1], idx[2], idx[3]) for idx in inds]
grad_field = [Interpolations.gradient(img_itp, idx[1], idx[2], idx[3]) for idx ∈ inds]
# hess_field = [Interpolations.hessian(img_itp, idx[1], idx[2], idx[3]) for idx ∈ inds]
hess_field = nothing # using the Gauss-Newton approximation

img_mov = extrapolate(interpolate(circshift(image, (10, 10, 10)), BSpline(Cubic())), Interpolations.Flat())

diff_vals = similar(inds, Float64)
fgh! = ImageRealign.make_fgh_function(vec(reference), img_mov, size(image) .÷ 2, inds, grad_field, hess_field, diff_vals)

G = zeros(6)
H = zeros(6,6)
p0 = zeros(6)

##
G_fd = grad(central_fdm(5, 1; factor=1e6), p -> fgh!(nothing, nothing, nothing, p), p0)[1]
fgh!(nothing, G, H, p0)
@test G ≈ G_fd rtol = 0.2

##
# function _grad(p)
#     G = zeros(6)
#     fgh!(nothing, G, nothing, p)
#     return G
# end
# H_fd = jacobian(central_fdm(5, 1; factor=1e6), _grad, p0)[1]
# fgh!(nothing, G, H, p0)
# @test H ≈ H_fd rtol = 0.5


## test translations
imgs = cat(image, circshift(image, (1, 0, 0)), circshift(image, (0, 1, 0)), circshift(image, (0, 0, 1)), dims=4)

p_ref = zeros(6, 4)
p_ref[4, 2] = 1
p_ref[5, 3] = 1
p_ref[6, 4] = 1

@test estimate_motion_parameters(imgs; ref_mode=1) ≈ p_ref rtol = 1e-1

p_ref[4, :] .-= 1
@test estimate_motion_parameters(imgs; ref_mode=2) ≈ p_ref rtol = 1e-1

p_ref[4, :] .+= 1
p_ref[5, :] .-= 1
@test estimate_motion_parameters(imgs; ref_mode=3) ≈ p_ref rtol = 1e-1

p_ref[5, :] .+= 1
p_ref[6, :] .-= 1
@test estimate_motion_parameters(imgs; ref_mode=4) ≈ p_ref rtol = 1e-1

## test free movement (rotaions and translations)
ps = [[0.1, 0, 0, 0, 0, 0],
      [0, 0.1, 0, 0, 0, 0],
      [0, 0, 0.1, 0, 0, 0],
      [0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 1],
]

img_interpolated = similar(image)
for p ∈ ps
    A = ImageRealign.create_affine_matrix(p, size(image) .÷ 2)

    for i ∈ CartesianIndices(image)
        v = A \ SVector{4,Float64}(i[1], i[2], i[3], 1)
        img_interpolated[i] = abs.(img_itp(v[1], v[2], v[3]))
    end
    @test estimate_motion_parameters(cat(image, img_interpolated; dims=4); ref_mode=1)[:,2] ≈ p atol = 1e-1
end
