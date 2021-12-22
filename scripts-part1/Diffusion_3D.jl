# run './mpiexecjl -n 4 julia --project scripts-part1/Diffusion_3D.jl' for multi-XPUs
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

@views norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))

@views function diffusion_3D(; do_visu=false)
    do_print = false
    # Physics
    Lx ,Ly ,Lz  = 10.0, 10.0, 10.0  # domain size
    Lxc,Lyc,Lzc = Lx/2, Ly/2, Lz/2  # domain center
    D           = 1.0               # diffusion coefficient
    ttot        = 1.0               # total simulation time
    dt          = 0.2               # physical time step

    # Numerics
    nx, ny, nz = 2^5, 2^5, 2^5      # local grid size    
    ncheck     = 100                # check residuals every ncheck pseudo time steps
    iter_max   = 1e6                # max pseudo time iterations
    tol        = 1e-8               # tolerance for converging

    # Derived numerics
    me, dims = init_global_grid(nx, ny, nz)                             # Initialization of MPI and more...
    @static if USE_GPU select_device() end                              # select one GPU per MPI local rank (if >1 GPU per node)
      dx,   dy,   dz = Lx/nx_g(), Ly/ny_g(), Lz/nz_g()                  # local grid steps
     _dx,  _dy,  _dz = 1.0/dx, 1.0/dy, 1.0/dz
    D_dx, D_dy, D_dz = D/dx, D/dy, D/dz
    nx_v, ny_v, nz_v = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]   # global grid size without halo
                  dτ = min(dx, dy, dz)^2/D/8.0                          # pseudo time step
                  nt = cld(ttot, dt)                                    # total number of physical time steps
    Xind, Yind, Zind = Int(cld(Lxc, dx)), Int(cld(Lyc, dy)), Int(cld(Lzc, dz)) 
    
    # Array initialisation
    H   = @zeros(nx, ny, nz)
    H  .= Data.Array([2.0*exp(-((x_g(ix,dx,H)+dx/2 -Lx/2)^2+(y_g(iy,dy,H)+dy/2 -Ly/2)^2+(z_g(iz,dz,H)+dz/2 -Lz/2)^2)/2) for ix=1:size(H,1), iy=1:size(H,2), iz=1:size(H,3)])
    Hτ  = copy(H)
    Hτ2 = copy(H)
    R_H = @zeros(nx-2, ny-2, nz-2)          # residual without halo
    size_H1_2, size_H2_2, size_H3_2 = size(H,1)-2, size(H,2)-2, size(H,3)-2

    # Prepare for 2D visualization
    if do_visu
        xc_v, yc_v = LinRange(dx+dx/2, Lx-dx-dx/2, nx_v), LinRange(dy+dy/2, Ly-dy-dy/2, ny_v)
        H_v   = zeros(nx_v, ny_v, nz_v)     # global H without boundaries (for visualization)
        H_inn = zeros(nx-2, ny-2, nz-2)     # no halo local array
        if (me==0) ENV["GKSwstype"]="nul"; if isdir("plots_out1")==false mkdir("plots_out1") end; loadpath = "./plots_out1/"; anim = Animation(loadpath,String[]); println("Animation directory: $(anim.dir)") end
        if (nx_v*ny_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
    end

    # Physical time stepping    
    for it = 1:nt   
        # update H using backward Euler
        H .= Hτ 

        t_tic = 0.0; niter = 0
        # pseudo time stepping
        for iter =1:iter_max  
            # start timer after warming
            if (iter==11) t_tic = Base.time(); niter = 0 end  
            # update Hτ in pseudo time
            @hide_communication (16, 2, 2) begin
                @parallel compute_pseudo!(Hτ2, Hτ, H, D_dx, D_dy, D_dz, dt, dτ, _dx, _dy, _dz, size_H1_2, size_H2_2, size_H3_2)
                Hτ, Hτ2 = Hτ2, Hτ   
                update_halo!(Hτ)
            end
            niter += 1
            # compute and check residuals
            if iter % ncheck == 0
                @parallel compute_residuals!(R_H, Hτ, H, D_dx, D_dy, D_dz, dt, _dx, _dy, _dz, size_H1_2, size_H2_2, size_H3_2)
                err = norm_g(Array(R_H))/(nx_v*ny_v*nz_v)
                if (me==0 && do_print) println(">> time = $(round(it*dt, sigdigits=3))s, iterations = $(iter), residual = $(round(err, sigdigits=3))") end
                if err < tol break; end
            end
        end

        # performance evaluation
        t_toc = Base.time() - t_tic
        A_eff = 3/1e9*nx_g()*ny_g()*nz_g()*sizeof(Float64)  # Effective main memory access per iteration [GB]
        t_it  = t_toc/niter                                 # Execution time per iteration [s]
        T_eff = A_eff/t_it                                  # Effective memory throughput [GB/s]
        if (me==0 && do_print) println("-----time = $(round(it*dt, sigdigits=3)) s, T_eff = $(round(T_eff, sigdigits=3)) GB/s-----") end
        
        # Visualization
        if do_visu && me==0
            H_inn .= Array(Hτ[2:end-1,2:end-1,2:end-1]); gather!(H_inn, H_v) 
            opts = (aspect_ratio=1, xlims=(xc_v[1], xc_v[end]), ylims=(yc_v[1], yc_v[end]),xtickfontsize=8,ytickfontsize=8,clims=(0, 1.5), xlabel="Lx", ylabel="Ly",xguidefontsize=8,yguidefontsize=8)
            P1 = heatmap(xc_v, yc_v, Array(H_v[:,:,nx÷2])'              ; opts..., title="0.5Lz (t=$(round(it*dt, sigdigits=3))s)", titlefontsize=10)
            P2 = heatmap(xc_v, yc_v, Array(H_v[:,:,Int(ceil(128*0.3))])'; opts..., title="0.3Lz", titlefontsize=10)
            P3 = heatmap(xc_v, yc_v, Array(H_v[:,:,Int(ceil(128*0.7))])'; opts..., title="0.7Lz", titlefontsize=10)
            display(plot(P2,P1,P3, layout = (1, 3), legend = false, size=(1500,400))); frame(anim)
        end
    end

    # Save animation
    if (do_visu && me==0) gif(anim, "diffusion_3D.gif", fps = 5)  end

    # Return the global array xc_g and H_g
    xc_g = LinRange(dx/2, Lx-dx/2, nx_g())
    H_g  = zeros(nx*dims[1],ny*dims[2],nz*dims[3])      
    if (me==0) gather!(Array(Hτ), H_g) end

    finalize_global_grid()
    return H_g, xc_g, H_g[Xind, Yind, Zind]
end

H_g, xc_g, Hc = diffusion_3D(; do_visu=false);