const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using Plots, Printf, ImplicitGlobalGrid
import MPI

# macros to avoid array allocation
macro qx(ix,iy,iz)  esc(:( -D_dx*(Hτ[$ix+1,$iy+1,$iz+1] - Hτ[$ix  ,$iy+1,$iz+1]) )) end
macro qy(ix,iy,iz)  esc(:( -D_dy*(Hτ[$ix+1,$iy+1,$iz+1] - Hτ[$ix+1,$iy  ,$iz+1]) )) end
macro qz(ix,iy,iz)  esc(:( -D_dz*(Hτ[$ix+1,$iy+1,$iz+1] - Hτ[$ix+1,$iy+1,$iz  ]) )) end

@parallel_indices (ix,iy,iz) function compute_pseudo!(Hτ2, Hτ, H, D_dx, D_dy, D_dz, dt, dτ, _dx, _dy, _dz, size_H1_2, size_H2_2, size_H3_2)
    if (ix<=size_H1_2 && iy<=size_H2_2 && iz<=size_H3_2)
        Hτ2[ix+1,iy+1,iz+1] = Hτ[ix+1,iy+1,iz+1] - dτ*( (Hτ[ix+1,iy+1,iz+1] - H[ix+1,iy+1,iz+1])/dt + (@qx(ix+1,iy,iz)-@qx(ix,iy,iz))*_dx + (@qy(ix,iy+1,iz)-@qy(ix,iy,iz))*_dy + (@qz(ix,iy,iz+1)-@qz(ix,iy,iz))*_dz )
    end
    return
end

@parallel_indices (ix,iy,iz) function compute_residuals!(R_H, Hτ, H, D_dx, D_dy, D_dz, dt, _dx, _dy, _dz, size_H1_2, size_H2_2, size_H3_2)
    if (ix<=size_H1_2 && iy<=size_H2_2 && iz<=size_H3_2)
        R_H[ix,iy,iz] = -(Hτ[ix+1,iy+1,iz+1] - H[ix+1,iy+1,iz+1])/dt - (@qx(ix+1,iy,iz)-@qx(ix,iy,iz))*_dx - (@qy(ix,iy+1,iz)-@qy(ix,iy,iz))*_dy - (@qz(ix,iy,iz+1)-@qz(ix,iy,iz))*_dz
    end
    return
end

@views function diffusion_3D(; do_visu=false)
    # Physics
    Lx, Ly, Lz = 10.0, 10.0, 10.0 # domain size
    D          = 1.0              # diffusion coefficient
    ttot       = 1.0              # total simulation time
    dt         = 0.2              # physical time step

    # Numerics
    nx, ny, nz = 32, 32, 32
    tol        = 1e-8

    # Derived numerics
    me, dims = init_global_grid(nx, ny, nz)  # Initialization of MPI and more...
    @static if USE_GPU select_device() end   # select one GPU per MPI local rank (if >1 GPU per node)
      dx,   dy,   dz = Lx/nx_g(), Ly/ny_g(), Lz/nz_g()
     _dx,  _dy,  _dz = 1.0/dx, 1.0/dy, 1.0/dz
    D_dx, D_dy, D_dz = D/dx, D/dy, D/dz
    nx_v, ny_v, nz_v = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]  
                  dτ = min(dx, dy, dz)^2/D/8.0
                  nt = cld(ttot, dt)
    
    # Array initialisation
    H   = @zeros(nx, ny, nz)
    H  .= Data.Array([2.0*exp(-((x_g(ix,dx,H)+dx/2 -Lx/2)^2+(y_g(iy,dy,H)+dy/2 -Ly/2)^2+(z_g(iz,dz,H)+dz/2 -Lz/2)^2)/2) for ix=1:size(H,1), iy=1:size(H,2), iz=1:size(H,3)])
    Hτ  = copy(H)
    Hτ2 = copy(H)
    size_H1_2, size_H2_2, size_H3_2 = size(H,1)-2, size(H,2)-2, size(H,3)-2
    R_H = @zeros(nx-2, ny-2, nz-2)  # residual without halo
    R_g = zeros(nx_v, ny_v, nz_v)   # global residual
    H_g = zeros(nx_v, ny_v, nz_v)   # global H without boundaries
    H_inn = zeros(nx-2, ny-2, nz-2) # no halo local array 

    # global coordinates: x, y, inner points only
    xc_g, yc_g = LinRange(dx+dx/2, Lx-dx-dx/2, nx_v), LinRange(dy+dy/2, Ly-dy-dy/2, ny_v) 

    # physical time stepping
    t_tic = 0.0; niter = 0
    for it = 1:nt 
        # if (it==11) t_tic = Base.time(); niter = 0 end
        # pseudo time stepping
        R_m = 1.0; iter = 0
        while R_m > tol
            @parallel compute_pseudo!(Hτ2, Hτ, H, D_dx, D_dy, D_dz, dt, dτ, _dx, _dy, _dz, size_H1_2, size_H2_2, size_H3_2)
            Hτ, Hτ2 = Hτ2, Hτ
            update_halo!(Hτ)
            # compute residuals
            @parallel compute_residuals!(R_H, Hτ, H, D_dx, D_dy, D_dz, dt, _dx, _dy, _dz, size_H1_2, size_H2_2, size_H3_2)
            gather!(Array(R_H), R_g) 
            R_m  = maximum(R_g) 
            iter += 1
            # print residuals
            # if iter % 20 == 0 && me == 0
            #     println("time = $(round(it*dt, sigdigits=3))s, iterations = $(iter), residual = $(R_m)")
            # end
        end
        # update H 
        H[:,:,:] = Hτ; niter += 1
        # gather global array
        H_inn .= Array(H[2:end-1,2:end-1,2:end-1]); gather!(H_inn, H_g)
        # Visualize
        if do_visu && me==0
            opts = (aspect_ratio=1, xlims=(xc_g[1], xc_g[end]), ylims=(yc_g[1], yc_g[end]), clims=(0.0, 1.0), c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
            display(heatmap(xc_g, yc_g, Array(H_g[:,:,15])'; opts...))
        end
    end

    # t_toc = Base.time() - t_tic
    # A_eff = 2/1e9*nx_g()*ny_g()*nz_g()*sizeof(Float64)  # Effective main memory access per iteration [GB]
    # t_it  = t_toc/niter                          # Execution time per iteration [s]
    # T_eff = A_eff/t_it                           # Effective memory throughput [GB/s]

    finalize_global_grid()
    return Array(H_g), xc_g
end

H_g, xc_g = diffusion_3D(; do_visu=false);