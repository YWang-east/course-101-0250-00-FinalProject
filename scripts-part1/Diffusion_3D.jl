const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, ImplicitGlobalGrid
import MPI

# macros to avoid array allocation
macro qx(ix,iy,iz)  esc(:( -D_dx*(C[$ix+1,$iy+1,$iz+1] - C[$ix  ,$iy+1,$iz+1]) )) end
macro qy(ix,iy,iz)  esc(:( -D_dy*(C[$ix+1,$iy+1,$iz+1] - C[$ix+1,$iy  ,$iz+1]) )) end
macro qz(ix,iy,iz)  esc(:( -D_dz*(C[$ix+1,$iy+1,$iz+1] - C[$ix+1,$iy+1,$iz  ]) )) end

@parallel_indices (ix,iy,iz) function compute_pseudo!(Hτ2, Hτ, H, D_dx, D_dy, D_dz, dt, dτ, _dx, _dy, _dz, size_H1_2, size_H2_2, size_H3_2)
    if (ix<=size_H1_2 && iy<=size_H2_2 && iz<=size_H3_2)
        Hτ2[ix+1,iy+1,iz+1] = Hτ[ix+1,iy+1,iz+1] - dτ*( (Hτ[ix+1,iy+1,iz+1] - H[ix+1,iy+1,iz+1])/dt 
                            + (@qx(ix+1,iy,iz)-@qx(ix,iy,iz))*_dx + (@qy(ix,iy+1,iz)-@qy(ix,iy,iz))*_dy + (@qz(ix,iy,iz+1)-@qz(ix,iy,iz))*_dz )
    end
    return
end

@parallel_indices (ix,iy,iz) function compute_physical!(H2, H, D_dx, D_dy, D_dz, dt,  _dx, _dy, _dz, size_H1_2, size_H2_2, size_H3_2)
if (ix<=size_H1_2 && iy<=size_H2_2 && iz<=size_H3_2)
    H2[ix+1,iy+1,iz+1] = H[ix+1,iy+1,iz+1] - dt*( (@qx(ix+1,iy) - @qx(ix,iy))*_dx + (@qy(ix,iy+1) - @qy(ix,iy))*_dy )
    return
end

@views function diffusion_3D(; do_visu=false)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    D          = 1.0              # diffusion coefficient
    ttot       = 1.0              # total simulation time
    dt         = 0.2              # physical time step

    # Numerics
    nx, ny, nz  = 32, 32, 32 
    nout    = 1

    # Derived numerics
    me, dims = init_global_grid(nx, ny, nz)  # Initialization of MPI and more...
    @static if USE_GPU select_device() end   # select one GPU per MPI local rank (if >1 GPU per node)

    dx, dy, dz       = Lx/nx_g(), Ly/ny_g(), Lz/nz_g()
    xc, yc, zc       = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny), LinRange(dz/2, Lz-dz/2, nz)
    _dx, _dy, _dz    = 1.0/dx, 1.0/dy, 1.0/dz
    D_dx, D_dy, D_dz = D/dx, D/dy, D/dz

    dτ = min(dx, dy, dz)^2/D/2.1
    nt = cld(ttot, dt)
    
    # Array initialisation
    H   = @zeros(nx, ny, nz)
    H  .= Data.Array([exp(-(x_g(ix,dx,H)+dx/2 -Lx/2)^2-(y_g(iy,dy,H)+dy/2 -Ly/2)^2-(z_g(iz,dz,H)+dz/2 -Lz/2)^2) for ix=1:size(H,1), iy=1:size(H,2), iz=1:size(H,3)])
    H2  = copy(H)
    Hτ  = copy(H)
    Hτ2  = copy(H)
    size_H1_2, size_H2_2, size_H3_2 = size(H,1)-2, size(H,2)-2, size(H,3)-2

    t_tic = 0.0; niter = 0
    # Dual-time stepping
    for it = 1:nt
        if (it==11) t_tic = Base.time(); niter = 0 end
        # @hide_communication (8, 2) begin
            @parallel compute!(C2, C, D_dx, D_dy, dt, _dx, _dy, size_C1_2, size_C2_2)
            C, C2 = C2, 
            update_halo!(C)
        # end
        niter += 1

        # Visualize
        if do_visu && (it % nout == 0)
            C_inn .= C[2:end-1,2:end-1]; gather!(C_inn, C_v)
            if (me==0)
                opts = (aspect_ratio=1, xlims=(Xi_g[1], Xi_g[end]), ylims=(Yi_g[1], Yi_g[end]), clims=(0.0, 1.0), c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
                heatmap(Xi_g, Yi_g, Array(C_v)'; opts...); frame(anim)
            end
        end
    end

    t_toc = Base.time() - t_tic
    A_eff = 2/1e9*nx_g()*ny_g()*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                          # Execution time per iteration [s]
    T_eff = A_eff/t_it                           # Effective memory throughput [GB/s]

    finalize_global_grid()
    return
end

diffusion_3D(; do_visu=true)