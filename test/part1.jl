# Testing part 1
using Test, ReferenceTests

# dir to the part 1 script
include("../scripts-part1/Diffusion_3D.jl") 

# Unit tests
Hτ = rand(5, 5, 5);
D_dx, D_dy, D_dz = 1.0, 1.0, 1.0;
@testset "Test heat flux macros" begin
    @test @qx(2,1,3) ≈ -(Hτ[3,2,4] - Hτ[2,2,4])
    @test @qx(4,3,1) ≈ -(Hτ[5,4,2] - Hτ[4,4,2])
    @test @qy(1,1,1) ≈ -(Hτ[2,2,2] - Hτ[2,1,2])
    @test @qy(3,2,3) ≈ -(Hτ[4,3,4] - Hτ[4,2,4])
    @test @qz(1,4,2) ≈ -(Hτ[2,5,3] - Hτ[2,5,2])
    @test @qz(2,3,4) ≈ -(Hτ[3,4,5] - Hτ[3,4,4]) 
end

# Reference test using ReferenceTests.jl
"Compare all dict entries"
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2; atol = 1e-5) for (v1,v2) in zip(values(d1), values(d2))])
inds = Int.(ceil.(LinRange(1, length(xc_g), 12)))
d = Dict(:X=> xc_g[inds], :H=>H_g[inds, inds, 15])

@testset "Reference Test" begin
    @test_reference "reftest-files/test_part_1.bson" d by=comp
end
