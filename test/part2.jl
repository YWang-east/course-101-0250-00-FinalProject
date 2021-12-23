# Testing part 2
using Test, ReferenceTests

# for testing, comment the function call: TM_2D(; do_visu = true) at the end of TM_2D_perf.jl
include("../scripts-part2/TM_2D_perf.jl")

# ------------------test macros defined in TM_2D_perf.jl---------------------
# Initialize arbitrary values for varialbes necessary for the macros
λb, θp, θv, θT = 1.0, 1.0, 1.0, 1.0
dx, dy = 1.0, 1.0
nx, ny = 5, 5                               # including ghost cells, choose small values for test speed
Vx, Vy = rand(nx-1,ny), rand(nx  ,ny-1)     
ηc, ηv = rand(nx  ,ny), rand(nx-1,ny-1)     
P , T  = rand(nx  ,ny), rand(nx  ,ny  )     
# Calculate variables using macros 
∇V   = [  @∇V(ix,iy) for ix=2:nx-1, iy=2:ny-1 ]  # inner cells, size (nx-2)(ny-2)  
ϵxx  = [ @ϵxx(ix,iy) for ix=2:nx-1, iy=2:ny-1 ]  # inner cells
ϵyy  = [ @ϵyy(ix,iy) for ix=2:nx-1, iy=2:ny-1 ]  # inner cells
ϵxy  = [ @ϵxy(ix,iy) for ix=1:nx-1, iy=1:ny-1 ]  # cell vertices, size (nx-1)(ny-1)
τxx  = [ @τxx(ix,iy) for ix=2:nx-1, iy=2:ny-1 ]  # inner cells
τyy  = [ @τyy(ix,iy) for ix=2:nx-1, iy=2:ny-1 ]  # inner cells
τxy  = [ @τxy(ix,iy) for ix=1:nx-1, iy=1:ny-1 ]  # cell vertices, size (nx-1)(ny-1)
dτP  = [ @dτP(ix,iy) for ix=2:nx-1, iy=2:ny-1 ]  # inner cells
dτVx = [@dτVx(ix,iy) for ix=2:nx-2, iy=2:ny-1 ]  # inner Vx, size (nx-3)(ny-2)
dτVy = [@dτVy(ix,iy) for ix=2:nx-1, iy=2:ny-2 ]  # inner Vy, size (nx-2)(ny-3)
dτT  = [ @dτT(ix,iy) for ix=2:nx-1, iy=2:ny-1 ]  # inner cells
# calculate variables using definitions (as references)
 ∇V_ref = diff(Vx[:,2:end-1],dims=1)./dx .+ diff(Vy[2:end-1,:],dims=2)./dy
ϵxx_ref = diff(Vx[:,2:end-1],dims=1)./dx .- 1/2 .* ∇V_ref
ϵyy_ref = diff(Vy[2:end-1,:],dims=2)./dy .- 1/2 .* ∇V_ref
ϵxy_ref = 0.5.*(diff(Vx,dims=2)./dy .+ diff(Vy,dims=1)./dx)
τxx_ref = 2 .* ηc[2:end-1,2:end-1] .*(ϵxx_ref .+ λb.*∇V_ref) .- P[2:end-1,2:end-1]
τyy_ref = 2 .* ηc[2:end-1,2:end-1] .*(ϵyy_ref .+ λb.*∇V_ref) .- P[2:end-1,2:end-1]
τxy_ref = 2 .* ηv .* ϵxy_ref

 dτP_ref = θp*4.1/max(nx-2,ny-2).*  ηc[2:end-1,2:end-1] .* (1.0+λb)
dτVx_ref = θv/4.1*(min(dx,dy)^2 ./( 0.5.*(ηc[2:end-2,2:end-1] + ηc[3:end-1,2:end-1]) ))./(1+λb)
dτVy_ref = θv/4.1*(min(dx,dy)^2 ./( 0.5.*(ηc[2:end-1,2:end-2] + ηc[2:end-1,3:end-1]) ))./(1+λb)
 dτT_ref = [θT/4.1* min(dx,dy)^2 for ix=2:nx-1, iy=2:ny-1] 
# Test macro calculations against reference
tol = 1e-10;
@testset "Test strain rate macros" begin
    @test isapprox(∇V ,  ∇V_ref, atol=tol)
    @test isapprox(ϵxx, ϵxx_ref, atol=tol)
    @test isapprox(ϵyy, ϵyy_ref, atol=tol)
    @test isapprox(ϵxy, ϵxy_ref, atol=tol)
end

@testset "Test stress macros" begin
    @test isapprox(τxx, τxx_ref, atol=tol)
    @test isapprox(τyy, τyy_ref, atol=tol)
    @test isapprox(τxy, τxy_ref, atol=tol)
end

@testset "Test pseudo time step macros" begin
    @test isapprox(dτP ,  dτP_ref, atol=tol)
    @test isapprox(dτVx, dτVx_ref, atol=tol)
    @test isapprox(dτVy, dτVy_ref, atol=tol)
    @test isapprox(dτT ,  dτT_ref, atol=tol)
end

