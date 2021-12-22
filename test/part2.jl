# Testing part 2
using Test, ReferenceTests

include("../scripts-part2/TM_2D_prototype.jl") 
include("../scripts-part2/TM_2D_perf.jl")

dVxdτ0_ref, dVydτ0_ref, ηc_ref, ηv_ref, dPdτ_ref, dVxdτ_ref, dVydτ_ref, Vx_ref, Vy_ref, P_ref, T_ref = TM_2D_baseline(; do_visu=false);
dVxdτ0, dVydτ0, ηc, ηv, dPdτ, dVxdτ, dVydτ, Vx, Vy, P, T = TM_2D(; do_visu=false);

# initial fields
# @test Vx[:,2:end-1] ≈ Vx_ref
# @test Vy[2:end-1,:] ≈ Vy_ref
# @test T[2:end-1,2:end-1] ≈ T_ref

# isapprox(dVxdτ[2:end-1,2:end-1],dVxdτ_ref,atol=1e-7)

# run 1 iterations
# @test ηc[2:end-1,2:end-1] ≈ ηc_ref
# @test ηv[2:end-1,2:end-1] ≈ ηv_ref[2:end-1,2:end-1]

# test macros
# dx, dy = 1.0, 1.0;
# nx, ny = 5, 5;
# Vx = [0 0 0 0 0;
#       1 1 1 1 1;
#       2 2 2 2 2;
#       3 3 3 3 3];

# Vy = [0 1 2 3;
#       0 1 2 3;
#       0 1 2 3;
#       0 1 2 3;
#       0 1 2 3];

# div_ref = diff(Vx[:,2:end-1],dims=1)./dx .+ diff(Vy[2:end-1,:],dims=2)./dy;
# div     = [@∇V(ix,iy) for ix=2:4, iy=2:4];