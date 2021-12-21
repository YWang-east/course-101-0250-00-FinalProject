# Testing part 2
using Test, ReferenceTests

include("../scripts-part2/TM_2D.jl") 
include("../scripts-part2/TM_2D_baseline.jl")

ηc_ref, ηv_ref, dVxdτ_ref, dVydτ_ref, Vx_ref, Vy_ref, P_ref, T_ref = TM_2D_baseline(; do_visu=false);
ηc, ηv, dVxdτ, dVydτ, Vx, Vy, P, T = TM_2D(; do_visu=false);

# Add unit and reference tests
# @test ϵii2 ≈ ϵii2_ref