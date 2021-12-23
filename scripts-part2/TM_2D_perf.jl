# Part 2: Thermomechanical coupling
const GPU_USE = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if GPU_USE
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, ImplicitGlobalGrid
import MPI

# macros to avoid array allocation
macro  ∇V(ix,iy)  esc(:(               (Vx[$ix,$iy]-Vx[$ix-1,$iy])/dx + (Vy[$ix,$iy]-Vy[$ix,$iy-1])/dy  )) end
macro ϵxx(ix,iy)  esc(:(         0.5*( (Vx[$ix,$iy]-Vx[$ix-1,$iy])/dx - (Vy[$ix,$iy]-Vy[$ix,$iy-1])/dy) )) end
macro τxx(ix,iy)  esc(:( ηc[$ix,$iy]*( (Vx[$ix,$iy]-Vx[$ix-1,$iy])/dx - (Vy[$ix,$iy]-Vy[$ix,$iy-1])/dy + 2*λb*( (Vx[$ix,$iy]-Vx[$ix-1,$iy])/dx + (Vy[$ix,$iy]-Vy[$ix,$iy-1])/dy )) - P[$ix,$iy] )) end
macro ϵyy(ix,iy)  esc(:(         0.5*(-(Vx[$ix,$iy]-Vx[$ix-1,$iy])/dx + (Vy[$ix,$iy]-Vy[$ix,$iy-1])/dy) )) end
macro τyy(ix,iy)  esc(:( ηc[$ix,$iy]*(-(Vx[$ix,$iy]-Vx[$ix-1,$iy])/dx + (Vy[$ix,$iy]-Vy[$ix,$iy-1])/dy + 2*λb*( (Vx[$ix,$iy]-Vx[$ix-1,$iy])/dx + (Vy[$ix,$iy]-Vy[$ix,$iy-1])/dy )) - P[$ix,$iy] )) end
macro ϵxy(ix,iy)  esc(:(         0.5*(( Vx[$ix,$iy+1]-Vx[$ix,$iy])/dy + (Vy[$ix+1,$iy]-Vy[$ix,$iy])/dx) )) end
macro τxy(ix,iy)  esc(:( ηv[$ix,$iy]*( (Vx[$ix,$iy+1]-Vx[$ix,$iy])/dy + (Vy[$ix+1,$iy]-Vy[$ix,$iy])/dx) )) end  
macro  qx(ix,iy)  esc(:( -(T[$ix+1,$iy+1] - T[$ix,$iy+1])/dx )) end
macro  qy(ix,iy)  esc(:( -(T[$ix+1,$iy+1] - T[$ix+1,$iy])/dy )) end
macro   S(ix,iy)  esc(:( 4* ηc[$ix,$iy]* ϵii2[$ix,$iy]^2 )) end
macro  η0(ix,iy)  esc(:( ϵii2[$ix,$iy]^(1/n-1)*exp( -T[$ix,$iy]*(1 /(1+T[$ix,$iy]/T0)) ) )) end

macro  dτP(ix,iy)  esc(:( θp*4.1/max(nx-2,ny-2)*ηc[$ix,$iy]*(1.0+λb) )) end
macro dτVx(ix,iy)  esc(:( θv/4.1*min(dx,dy)^2 /((ηc[$ix,$iy]+ηc[$ix+1,$iy])/2)/(1.0+λb) )) end
macro dτVy(ix,iy)  esc(:( θv/4.1*min(dx,dy)^2 /((ηc[$ix,$iy]+ηc[$ix,$iy+1])/2)/(1.0+λb) )) end
macro  dτT(ix,iy)  esc(:( θT/4.1*min(dx,dy)^2 )) end
"""
  compute the second order invariant of the deviatoric strain rate tensor
"""
@parallel_indices (ix,iy) function compute_2invar!(ϵii2, Vx, Vy, dx, dy, nx, ny)
  if ( ix<=(nx-2) && iy<=(ny-2) ) 
    ϵii2[ix+1,iy+1] = ( 0.5*(@ϵxx(ix+1,iy+1)^2 + @ϵyy(ix+1,iy+1)^2) + (0.25*(@ϵxy(ix,iy) + @ϵxy(ix+1,iy) + @ϵxy(ix,iy+1) + @ϵxy(ix+1,iy+1)))^2 )^0.5 
  end
  return
end
"""
  compute nonlinear viscosity defined at cell centers
"""
@parallel_indices (ix,iy) function compute_viscosity_c!(ηc, ϵii2, n, T, T0, θη, nx, ny)
  if ( ix<=(nx-2) && iy<=(ny-2) ) 
    ηc[ix+1,iy+1] = exp( θη*log(ηc[ix+1,iy+1]) + (1-θη)*log(@η0(ix+1,iy+1)) ) 
  end
  return
end
"""
  compute nonlinear viscosity at cell vertices by averaging the values of 4 neighboring cell centers
"""
@parallel_indices (ix,iy) function compute_viscosity_v!(ηv, ηc, size_ηv1, size_ηv2)
  if ( ix<=size_ηv1 && iy<=size_ηv2 ) 
    ηv[ix,iy] = 0.25*( ηc[ix,iy] + ηc[ix+1,iy] + ηc[ix,iy+1] + ηc[ix+1,iy+1] ) 
  end
  return
end
"""
  compute residuals of continuity and temperature equations
"""
@parallel_indices (ix,iy) function compute_residuals_P_T!(dPdτ, dTdτ, P, Vx, Vy, T, Tt, ηc, ϵii2, n, T0, dt, dx, dy, nx, ny)
  if ( ix<=(nx-2) && iy<=(ny-2) )
    dPdτ[ix+1,iy+1] = - @∇V(ix+1,iy+1)
    dTdτ[ix+1,iy+1] = (Tt[ix+1,iy+1]-T[ix+1,iy+1])/dt - (@qx(ix+1,iy)-@qx(ix,iy))/dx - (@qy(ix,iy+1)-@qy(ix,iy))/dy + @S(ix+1,iy+1)  
  end
  return
end
"""
  compute residuals of x y momentum equations, including damped residuals
"""
@parallel_indices (ix,iy) function compute_residuals_Vx_Vy!(dVxdτ, dVydτ, dVxdτ0, dVydτ0, P, Vx, Vy, ηv, ηc, λb, dampx, dampy, dx, dy, size_Vx1, size_Vx2, size_Vy1, size_Vy2)
  if ( ix<=(size_Vx1-2) && iy<=(size_Vx2-2) )
    dVxdτ0[ix+1,iy+1] = dVxdτ[ix+1,iy+1] + dampx * dVxdτ0[ix+1,iy+1] 
     dVxdτ[ix+1,iy+1] = (@τxy(ix+1,iy+1) - @τxy(ix+1,iy))/dy + (@τxx(ix+2,iy+1) - @τxx(ix+1,iy+1))/dx end
  if ( ix<=(size_Vy1-2) && iy<=(size_Vy2-2) ) 
    dVydτ0[ix+1,iy+1] = dVydτ[ix+1,iy+1] + dampy * dVydτ0[ix+1,iy+1]
     dVydτ[ix+1,iy+1] = (@τxy(ix+1,iy+1) - @τxy(ix,iy+1))/dx + (@τyy(ix+1,iy+2) - @τyy(ix+1,iy+1))/dy end
  return
end
"""
  update pressure and temperature in pseudo time
"""
@parallel_indices (ix,iy) function stepping_pseudo_P_T!(P, T, dPdτ, dTdτ, ηc, θp, θT, λb, dx, dy, nx, ny)
  if ( ix<=(nx-2) && iy<=(ny-2) )
    P[ix+1,iy+1] = P[ix+1,iy+1] + @dτP(ix+1,iy+1)*dPdτ[ix+1,iy+1]
    T[ix+1,iy+1] = T[ix+1,iy+1] + @dτT(ix+1,iy+1)*dTdτ[ix+1,iy+1] 
  end
  return
end  
"""
  update velocities in pseudo time
"""
@parallel_indices (ix,iy) function stepping_pseudo_Vx_Vy!(Vx, Vy, dVxdτ, dVydτ, dVxdτ0, dVydτ0, ηc, λb, θv, dampx, dampy, dx, dy, size_Vx1, size_Vx2, size_Vy1, size_Vy2)
  if ( ix<=(size_Vx1-2) && iy<=(size_Vx2-2) ) Vx[ix+1,iy+1] = Vx[ix+1,iy+1] + @dτVx(ix+1,iy+1)*( dVxdτ[ix+1,iy+1]+ dampx*dVxdτ0[ix+1,iy+1] )  end
  if ( ix<=(size_Vy1-2) && iy<=(size_Vy2-2) ) Vy[ix+1,iy+1] = Vy[ix+1,iy+1] + @dτVy(ix+1,iy+1)*( dVydτ[ix+1,iy+1]+ dampy*dVydτ0[ix+1,iy+1] ) end
  return
end
"""
  compute the norm of a matrix
"""
@views norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))

@views function TM_2D(; do_visu = false, do_print = false)
  # Physics
  Lx, Ly = 0.86038, 0.86038     # global domain size 
  n      = 3                    # power law rheology exponent
  Vbc    = 66.4437              # boundary velocity  
  T0     = 49.3269 / n          # initial temperature
  r      = 0.0737               # perturbation radius
  Tamp   = 0.1*T0               # perturbation temperature

  # Numerics
  nx, ny = 32*2^2, 32*2^2       # local grid size
  tol    = 1e-5                 # non-linear tolerance
  ξ      = 10.0                 # physical steps reduction for temperature
  λb     = 1.0                  # numerical bulk viscosity
  θη     = 0.9                  # relaxation factor for viscosity
  ν      = 4.0                  # damping factor for velocity residuals
  nt     = 5                    # number of time steps
  iter_max = 1e3                # max PT iterations
  n_check  = 2e2                # residuals checking step
  n_vis    = 1                  # visualization step
  θp,θv,θT = 0.5, 0.5, 0.5      # PT steps reduction for pressure, velocities and temperature 
  
  # Derived numerics
  me, dims = init_global_grid(nx, ny, 1)  # Initialize 2D MPI 
  @static if GPU_USE select_device() end  # select one GPU per MPI local rank (if >1 GPU per node)
  dx, dy = Lx/(nx_g()-2), Ly/(ny_g()-2)   # local grid step
  dt     = ξ*min(dx, dy)^2/4.1            # physical time step  
  dampx  = 1 - ν/(nx-2)                   # damping for x-momentum residuals
  dampy  = 1 - ν/(ny-2)                   # damping for y-momentum residuals

  # Array initialisation
  P      = @zeros(nx  , ny  )    # pressure 
  T      = @zeros(nx  , ny  )    # temperature
  Tt     = @zeros(nx  , ny  )    # temperature at current physical time step
  ηc     =  @ones(nx  , ny  )    # effective viscosity at cell centroids
  ηv     =  @ones(nx-1, ny-1)    # effective viscosity at cell vertices
  Vx     = @zeros(nx-1, ny  )    # x-velocity 
  Vy     = @zeros(nx  , ny-1)    # y-velocity
  ϵii2   = @zeros(nx  , ny  )    # 2nd order invariant
  dPdτ   = copy(P)               # continuity residuals
  dTdτ   = copy(T)               # temperature residuals
  dVxdτ  = copy(Vx)              # x-momentum residuals
  dVydτ  = copy(Vy)              # y-momentum residuals
  dVxdτ0 = copy(dVxdτ)           # damped x-momentum residuals
  dVydτ0 = copy(dVydτ)           # damped y-momentum residuals
  size_Vx1, size_Vx2 = size(Vx,1), size(Vx,2)
  size_Vy1, size_Vy2 = size(Vy,1), size(Vy,2)
  size_ηv1, size_ηv2 = size(ηv,1), size(ηv,2)

  # Initial conditions
  Vx[:,2:end-1] .= Data.Array([ Vbc*(x_g(ix,dx,Vx)-dx/2)/Lx for ix=1:size(Vx,1), iy=2:size(Vx,2)-1])
  Vy[2:end-1,:] .= Data.Array([-Vbc*(y_g(iy,dy,Vy)-dy/2)/Ly for ix=2:size(Vy,1)-1, iy=1:size(Vy,2)])
  Vx[:,[1 end]] .= Vx[:,[2 end-1]]
  Vy[[1 end],:] .= Vy[[2 end-1],:]
  T[2:end-1,2:end-1] .= Data.Array([(x_g(ix,dx,T)-dx/2)^2 + (y_g(iy,dy,T)-dy/2)^2 < r^2 for ix=2:nx-1, iy=2:ny-1] .* Tamp)
  T[:, [1 end]] .= T[:,[2 end-1]] 
  T[[1 end], :] .= T[[2 end-1],:] 

  # Preparation of visualisation
  if do_visu
    if (me==0) ENV["GKSwstype"]="nul"; if isdir("TM2D_out")==false mkdir("TM2D_out") end; loadpath = "./TM2D_out/"; anim = Animation(loadpath,String[]); println("Animation directory: $(anim.dir)") end
    nx_v, ny_v = (nx-2)*dims[1], (ny-2)*dims[2]
    if (nx_v*ny_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
    T_v   = zeros(nx_v, ny_v) # global array for visu
    T_inn = zeros(nx-2, ny-2) # no halo local array for visu
    xc, yc = LinRange(dx/2, Lx-dx/2, nx_v), LinRange(dy/2, Ly-dy/2, ny_v) # inner cell centers
  end

  # Physical time stepping
  for it = 1:nt 
    Tt .= T     # temperature updates from previous timestep
    t_tic = 0.0; niter = 0
    for iter = 1:iter_max # pseudo transient stepping
      # start timer after warming up
      if (iter==11) t_tic = Base.time(); niter = 0 end

      # compute 2nd order invariant of the deviatoric strain rate tensor
      @parallel compute_2invar!(ϵii2, Vx, Vy, dx, dy, nx, ny)

      # compute viscosity at cell centers and vertices
      @parallel compute_viscosity_c!(ηc, ϵii2, n, T, T0, θη, nx, ny) 
      update_halo!(ηc)
      @parallel compute_viscosity_v!(ηv, ηc, size_ηv1, size_ηv2)

      # compute residuals for inner cells
      @parallel compute_residuals_Vx_Vy!(dVxdτ, dVydτ, dVxdτ0, dVydτ0, P, Vx, Vy, ηv, ηc, λb, dampx, dampy, dx, dy, size_Vx1, size_Vx2, size_Vy1, size_Vy2)
      @parallel compute_residuals_P_T!(dPdτ, dTdτ, P, Vx, Vy, T, Tt, ηc, ϵii2, n, T0, dt, dx, dy, nx, ny)
      
      # update velocities, pressure and temperature
      @parallel stepping_pseudo_Vx_Vy!(Vx, Vy, dVxdτ, dVydτ, dVxdτ0, dVydτ0, ηc, λb, θv, dampx, dampy, dx, dy, size_Vx1, size_Vx2, size_Vy1, size_Vy2) # inner cells
      @parallel stepping_pseudo_P_T!(P, T, dPdτ, dTdτ, ηc, θp, θT, λb, dx, dy, nx, ny)  
       T[:, [1 end]] .=  T[:,[2 end-1]] # zero heat flux at top and bottom boundaries
       T[[1 end], :] .=  T[[2 end-1],:] # zero heat flux at left and right boundaries
      Vx[:, [1 end]] .= Vx[:,[2 end-1]] # zero shear stress at top and bottom boundaries
      Vy[[1 end], :] .= Vy[[2 end-1],:] # zero shear stress at left and right boundaries
      update_halo!(T, Vx, Vy)
      niter += 1

      # check residuals
      if iter % n_check == 0 
        R_P  = norm_g( Array( dPdτ[2:end,2:end]) )/( (nx_g()-2)*(ny_g()-2) )
        R_Vx = norm_g( Array(dVxdτ[2:end,2:end]) )/( (nx_g()-2)*(ny_g()-2) )
        R_Vy = norm_g( Array(dVydτ[2:end,2:end]) )/( (nx_g()-2)*(ny_g()-2) )
        R_T  = norm_g( Array( dTdτ[2:end,2:end]) )/( (nx_g()-2)*(ny_g()-2) )
        if (me==0 && do_print) println(">>time = $(round(it*dt, sigdigits=3)), iter = $(iter), R_P = $(round(R_P,sigdigits=3)), R_Vx = $(round(R_Vx,sigdigits=3)), R_Vy = $(round(R_Vy,sigdigits=3)), R_T = $(round(R_T,sigdigits=3))") end
        if max(R_P, R_Vx, R_Vy, R_T) < tol break; end
      end  
    end # end pseudo transient stepping

    # performance evaluation
    t_toc = Base.time() - t_tic
    A_eff = 10/1e9*nx_g()*ny_g()*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                           # Execution time per iteration [s]
    T_eff = A_eff/t_it                            # Effective memory throughput [GB/s]
    if (me==0) println("-----time = $(round(it*dt, sigdigits=3)) s, T_eff = $(round(T_eff, sigdigits=3)) GB/s-----") end

    # visualization
    if do_visu && (it%n_vis == 0) && me==0
      T_inn .= Array(T[2:end-1,2:end-1]); gather!(T_inn, T_v)
      opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
      display(heatmap(xc, yc, Array(T_v)'; opts...)); frame(anim)
    end

  end # end physical stepping 

  # Save animation
  if (do_visu && me==0) gif(anim, "TM_2D.gif", fps = 5)  end

  finalize_global_grid()
  return
end

TM_2D(; do_visu = false, do_print = false)
