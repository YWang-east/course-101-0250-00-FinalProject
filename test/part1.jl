# Testing part 1
using Test, ReferenceTests

# dir to the part 1 script
include("../scripts-part1/Diffusion_3D.jl") 

# Reference test using ReferenceTests.jl
"Compare all dict entries"
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2; atol = 1e-5) for (v1,v2) in zip(values(d1), values(d2))])
inds = Int.(ceil.(LinRange(1, length(xc_g), 12)))
d = Dict(:X=> xc_g[inds], :H=>H_g[inds, inds, 15])

@testset "Reference Test" begin
    @test_reference "reftest-files/test_part_1.bson" d by=comp
end
