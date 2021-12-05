const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, ImplicitGlobalGrid
import MPI

# macros to avoid array allocation
macro qx(ix,iy)  esc(:( -D_dx*(H[$ix+1,$iy+1] - H[$ix  ,$iy+1]) )) end
macro qy(ix,iy)  esc(:( -D_dy*(H[$ix+1,$iy+1] - H[$ix+1,$iy  ]) )) end
macro _qx(ix,iy)  esc(:( -D_dx*(Hτ[$ix+1,$iy+1] - Hτ[$ix  ,$iy+1]) )) end
macro _qy(ix,iy)  esc(:( -D_dy*(Hτ[$ix+1,$iy+1] - Hτ[$ix+1,$iy  ]) )) end

@parallel_indices (ix,iy) function compute_pseudo!(Hτ2, Hτ, H, D_dx, D_dy, dt, dτ, _dx, _dy, size_H1_2, size_H2_2)
    if (ix<=size_H1_2 && iy<=size_H2_2)
        Hτ2[ix+1,iy+1] = Hτ[ix+1,iy+1] - dτ*( (Hτ[ix+1,iy+1] - H[ix+1,iy+1])/dt + (@qx(ix+1,iy)-@qx(ix,iy))*_dx + (@qy(ix,iy+1)-@qy(ix,iy))*_dy )
    end
    return
end

@parallel_indices (ix,iy) function compute_residuals!(R_H, Hτ, H, D_dx, D_dy, dt, _dx, _dy, size_H1_2, size_H2_2)
    if (ix<=size_H1_2 && iy<=size_H2_2)
        R_H[ix,iy] = -(Hτ[ix+1,iy+1] - H[ix+1,iy+1])/dt - (@qx(ix+1,iy)-@qx(ix,iy))*_dx - (@qy(ix,iy+1)-@qy(ix,iy))*_dy 
    end
    return
end

@parallel_indices (ix,iy) function compute_physical!(H2, H, Hτ, D_dx, D_dy, dt, _dx, _dy, size_H1_2, size_H2_2)
    if (ix<=size_H1_2 && iy<=size_H2_2)
        H2[ix+1,iy+1] = H[ix+1,iy+1] - dt*( (@_qx(ix+1,iy) - @_qx(ix,iy))*_dx + (@_qy(ix,iy+1) - @_qy(ix,iy))*_dy )
    end
    return
end


@views function diffusion_2D(; do_visu=false)
    # Physics
    Lx, Ly = 10.0, 10.0 # domain size
    D      = 1.0        # diffusion coefficient
    ttot   = 1.0        # total simulation time
    dt     = 0.2        # physical time step

    # Numerics
    nx, ny = 32, 32
    nout   = 1
    tol    = 1e-8

    # Derived numerics
    me, dims = init_global_grid(nx, ny, 1)  # Initialization of MPI and more...
    @static if USE_GPU select_device() end   # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy     = Lx/nx_g(), Ly/ny_g()
    xc, yc     = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    _dx, _dy   = 1.0/dx, 1.0/dy
    D_dx, D_dy = D/dx, D/dy
    dτ         = min(dx, dy)^2/D/2.1
    nt         = cld(ttot, dt)
    nx_v, ny_v = (nx-2)*dims[1], (ny-2)*dims[2] # global dimensions

    # Array initialisation
    H   = @zeros(nx, ny)
    H  .= Data.Array([exp(-(x_g(ix,dx,H)+dx/2 -Lx/2)^2-(y_g(iy,dy,H)+dy/2 -Ly/2)^2) for ix=1:size(H,1), iy=1:size(H,2)])
    H2  = copy(H)
    Hτ  = copy(H)
    Hτ2 = copy(H)
    size_H1_2, size_H2_2 = size(H,1)-2, size(H,2)-2
    R_H = @ones(nx-2, ny-2) # residual without halo
    R_v = ones(nx_v, ny_v)  # global residual
    R_m = maximum(R_v)      # maximum residual
    err = []

    # Preparation of visualisation
    if do_visu
        H_v   = zeros(nx_v, ny_v) # global array for visu
        H_inn = zeros(nx-2, ny-2) # no halo local array for visu
        Xi_g, Yi_g = LinRange(dx+dx/2, Lx-dx-dx/2, nx_v), LinRange(dy+dy/2, Ly-dy-dy/2, ny_v) # inner points only
    end

    t_tic = 0.0; niter = 0
    # physical time stepping
    for it = 1:nt 
        if (it==11) t_tic = Base.time(); niter = 0 end
        # pseudo time stepping
        while R_m > tol
            @parallel compute_pseudo!(Hτ2, Hτ, H, D_dx, D_dy, dt, dτ, _dx, _dy, size_H1_2, size_H2_2)
            Hτ, Hτ2 = Hτ2, Hτ
            update_halo!(Hτ)
            # compute residuals
            @parallel compute_residuals!(R_H, Hτ, H, D_dx, D_dy, dt, _dx, _dy, size_H1_2, size_H2_2); gather!(R_H, R_v) 
            R_m = maximum(R_v)
        end

        @parallel compute_physical!(H2, H, Hτ, D_dx, D_dy, dt, _dx, _dy, size_H1_2, size_H2_2)
        H, H2 = H2, H
        update_halo!(H)

        niter += 1
        # Visualize
        if do_visu && (it % nout == 0)
            H_inn .= H[2:end-1,2:end-1]; gather!(H_inn, H_v)
            if (me==0)
                opts = (aspect_ratio=1, xlims=(Xi_g[1], Xi_g[end]), ylims=(Yi_g[1], Yi_g[end]), clims=(0.0, 1.0), c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
                heatmap(Xi_g, Yi_g, Array(H_v)'; opts...)
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

diffusion_2D(; do_visu=true)