# Part 2: Thermomechanical coupling
# const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
# @static if USE_GPU
#     @init_parallel_stencil(CUDA, Float64, 2)
# else
#     @init_parallel_stencil(Threads, Float64, 2)
# end
using Plots, Printf, ImplicitGlobalGrid
import MPI

# macros to avoid array allocation
macro  ∇V(ix,iy)  esc(:(               (Vx[$ix+1,$iy]-Vx[$ix,$iy  ])/dx + (Vy[$ix,$iy+1]-Vy[$ix  ,$iy])/dy  )) end
macro ϵxx(ix,iy)  esc(:(         0.5*( (Vx[$ix+1,$iy]-Vx[$ix,$iy  ])/dx - (Vy[$ix,$iy+1]-Vy[$ix  ,$iy])/dy) )) end
macro τxx(ix,iy)  esc(:( ηc[$ix,$iy]*( (Vx[$ix+1,$iy]-Vx[$ix,$iy  ])/dx - (Vy[$ix,$iy+1]-Vy[$ix  ,$iy])/dy + 2*λb*(Vx[$ix+1,$iy]-Vx[$ix,$iy])/dx + (Vy[$ix,$iy+1]-Vy[$ix,$iy])/dy) - P[$ix,$iy] )) end
macro ϵyy(ix,iy)  esc(:(         0.5*(-(Vx[$ix+1,$iy]-Vx[$ix,$iy  ])/dx + (Vy[$ix,$iy+1]-Vy[$ix  ,$iy])/dy) )) end
macro τyy(ix,iy)  esc(:( ηc[$ix,$iy]*(-(Vx[$ix+1,$iy]-Vx[$ix,$iy  ])/dx + (Vy[$ix,$iy+1]-Vy[$ix  ,$iy])/dy + 2*λb*(Vx[$ix+1,$iy]-Vx[$ix,$iy])/dx + (Vy[$ix,$iy+1]-Vy[$ix,$iy])/dy) - P[$ix,$iy] )) end
macro ϵxy(ix,iy)  esc(:(         0.5*(  Vx[$ix  ,$iy]-Vx[$ix,$iy-1])/dy + (Vy[$ix,$iy  ]-Vy[$ix-1,$iy])/dx  )) end
macro τxy(ix,iy)  esc(:( ηv[$ix,$iy]*( (Vx[$ix  ,$iy]-Vx[$ix,$iy-1])/dy + (Vy[$ix,$iy  ]-Vy[$ix-1,$iy])/dx) )) end  
macro  qx(ix,iy)  esc(:( -(T[$ix,$iy] - T[$ix-1,$iy  ])/dx )) end
macro  qy(ix,iy)  esc(:( -(T[$ix,$iy] - T[$ix  ,$iy-1])/dy )) end
macro   S(ix,iy)  esc(:( 4* ηc[$ix,$iy]* ϵii2[$ix,$iy]^2 )) end
macro  η0(ix,iy)  esc(:( ϵii2[$ix,$iy]^(1/n-1)*exp( -T[$ix,$iy]*(1 /(1+T[$ix,$iy]/T0)) ) )) end

macro  dτP(ix,iy)  esc(:( θp*4.1/max(nx,ny)*ηc[$ix,$iy]*(1.0+λb) )) end
macro dτVx(ix,iy)  esc(:( θv/4.1*min(dx,dy)^2 /(ηc[$ix,$iy]+ηc[$ix+1,$iy])/2/(1.0+λb) )) end
macro dτVy(ix,iy)  esc(:( θv/4.1*min(dx,dy)^2 /(ηc[$ix,$iy]+ηc[$ix,$iy+1])/2/(1.0+λb) )) end
macro  dτT(ix,iy)  esc(:( θT/4.1*min(dx,dy)^2 )) end


@parallel_indices (ix,iy) function compute_2invar!(ϵii2, Vx, Vy, dx, dy, nx, ny)
  if     (ix==1  && iy==1 ) ϵii2[ix,iy] = ( 0.5*(@ϵxx(ix,iy)^2 + @ϵyy(ix,iy)^2) + (0.25*@ϵxy(ix+1,iy+1))^2 )^0.5  # bottom-left corner
  elseif (ix==nx && iy==1 ) ϵii2[ix,iy] = ( 0.5*(@ϵxx(ix,iy)^2 + @ϵyy(ix,iy)^2) + (0.25*@ϵxy(ix  ,iy+1))^2 )^0.5  # bottom-right corner
  elseif (ix==1  && iy==ny) ϵii2[ix,iy] = ( 0.5*(@ϵxx(ix,iy)^2 + @ϵyy(ix,iy)^2) + (0.25*@ϵxy(ix+1,iy  ))^2 )^0.5  # top-left corner
  elseif (ix==nx && iy==ny) ϵii2[ix,iy] = ( 0.5*(@ϵxx(ix,iy)^2 + @ϵyy(ix,iy)^2) + (0.25*@ϵxy(ix  ,iy  ))^2 )^0.5  # top-right corner 
  elseif (ix==1  && iy>1 && iy<ny) ϵii2[ix,iy] = ( 0.5*(@ϵxx(ix,iy)^2 + @ϵyy(ix,iy)^2) + (0.25*(@ϵxy(ix+1,iy  )+@ϵxy(ix+1,iy+1)))^2 )^0.5   # left edge
  elseif (ix==nx && iy>1 && iy<ny) ϵii2[ix,iy] = ( 0.5*(@ϵxx(ix,iy)^2 + @ϵyy(ix,iy)^2) + (0.25*(@ϵxy(ix  ,iy  )+@ϵxy(ix  ,iy+1)))^2 )^0.5   # right edge
  elseif (iy==1  && ix>1 && ix<nx) ϵii2[ix,iy] = ( 0.5*(@ϵxx(ix,iy)^2 + @ϵyy(ix,iy)^2) + (0.25*(@ϵxy(ix  ,iy+1)+@ϵxy(ix+1,iy+1)))^2 )^0.5   # bottom edge
  elseif (iy==ny && ix>1 && ix<nx) ϵii2[ix,iy] = ( 0.5*(@ϵxx(ix,iy)^2 + @ϵyy(ix,iy)^2) + (0.25*(@ϵxy(ix  ,iy  )+@ϵxy(ix+1,iy  )))^2 )^0.5   # top edge
  else ϵii2[ix,iy] = ( 0.5*(@ϵxx(ix,iy)^2 + @ϵyy(ix,iy)^2) + (0.25*(@ϵxy(ix,iy) + @ϵxy(ix+1,iy) + @ϵxy(ix,iy+1) + @ϵxy(ix+1,iy+1)))^2 )^0.5 # inner points
  end
  return
end

@parallel_indices (ix,iy) function compute_viscosity!(ηc, ηv, ϵii2, n, T, T0, θη)
  if (ix<=nx && iy<=ny) 
    ηc[ix,iy] = exp( θη*log(ηc[ix,iy]) + (1-θη)*log(@η0(ix,iy)) ) 
    if (ix>1 && iy>1)  ηv[ix,iy] = 0.25*( ηc[ix-1,iy-1] + ηc[ix-1,iy] + ηc[ix,iy-1] + ηc[ix,iy] ) end
  end
  return
end

@parallel_indices (ix,iy) function compute_residuals!(dPdτ, dVxdτ, dVydτ, dTdτ, P, Vx, Vy, T, Tt, ηc, ηv, ϵii2, n, T0, θp, θv, θT, θη, λb)
  if (ix<=nx && iy<=ny)
    
  else
     dPdτ[ix,iy] = - @∇V(ix,iy)
    dVxdτ[ix,iy] = (@τxy(ix+1,iy+1)-@τxy(ix+1,iy  ))/dy + (@τxx(ix+1,iy  )-@τxx(ix,iy))/dx
    dVydτ[ix,iy] = (@τxy(ix+1,iy+1)-@τxy(ix  ,iy+1))/dx + (@τyy(ix  ,iy+1)-@τyy(ix,iy))/dy
     dTdτ[ix,iy] = (Tt[ix,iy]-T[ix,iy])/dt - (@qx(ix+1,iy)-@qx(ix,iy))/dx - (@qy(ix,iy+1)-@qy(ix,iy))/dy + @S(ix,iy)
  end
  return
end

@views function TM_2D(; do_visu = false)
  # Physics
  Lx, Ly = 0.86038, 0.86038     # domain size 
  n      = 3                    # power law rheology exponent
  Vbc    = 66.4437              # boundary velocity  
  T0     = 49.3269 / n          # initial temperature
  r      = 0.0737               # perturbation radius
  Tamp   = 0.1*T0               # perturbation temperature

  # Numerics
  nx, ny = 32*2^2, 32*2^2       # grid size
  dx, dy = Lx/nx, Ly/ny         # gris step
  tol    = 1e-5                 # non-linear tolerance
  ξ      = 10.0                 # physical steps reduction for temperature
  θp,θv,θT = 0.5, 0.5, 0.5                  # PT steps reduction for pressure, velocities and temperature 
  λb     = 1.0                  # numerical bulk viscosity
  θη     = 0.9                  # relaxation factor for viscosity
  ν      = 4.0                  # damping factor for velocity residuals
  nt     = 54                   # number of time steps
  nout   = 100                  # residuals checking step
  n_vis  = 2                    # visualization step
  iter_max = 1e6                # max PT iterations
  
  # Derived numerics
  dt    = ξ*min(dx, dy)^2/4.1   # physical time step  
  dampx = 1 - ν/nx              # damping for x-momentum residuals
  dampy = 1 - ν/ny              # damping for y-momentum residuals
    
  # Grid: staggered
    xn,   yn = LinRange(0   , Lx     , nx+1), LinRange(0   , Ly     , ny+1)   # cell interface
    xc,   yc = LinRange(dx/2, Lx-dx/2, nx  ), LinRange(dy/2, Ly-dy/2, ny  )   # cell center
    Xc,   Yc = repeat(xc, 1, ny  ), repeat(yc', nx  , 1)                      # location of scalar quantities
  X_fx, Y_fx = repeat(xn, 1, ny  ), repeat(yc', nx+1, 1)                      # location of x-fluxes  
  X_fy, Y_fy = repeat(xc, 1, ny+1), repeat(yn', nx  , 1)                      # location of y-fluxes

  # Array initialisation
  P      = zeros(Float64, nx  , ny  )   # pressure 
  T      = zeros(Float64, nx  , ny  )   # temperature
  Tt     = zeros(Float64, nx  , ny  )   # temperature at current physical time step
  ηc     =  ones(Float64, nx  , ny  )   # effective viscosity at cell centroids
  ηv     =  ones(Float64, nx+1, ny+1)   # effective viscosity at cell vertices
  Vx     = zeros(Float64, nx+1, ny  )   # x-velocity 
  Vy     = zeros(Float64, nx  , ny+1)   # y-velocity
  dVxdτ  = zeros(Float64, nx-1, ny  )   # x-momentum residuals
  dVydτ  = zeros(Float64, nx  , ny-1)   # y-momentum residuals
  dVxdτ0 = copy(dVxdτ)
  dVydτ0 = copy(dVydτ)
    
  # Initial conditions
  Vx  .=  Vbc.*X_fx./Lx
  Vy  .= -Vbc.*Y_fy./Ly
  T[(Xc.^2 .+ Yc.^2) .< r^2] .= Tamp

  for it = 1:nt # Physical stepping
    Tt .= T     # temperature updates from previous timestep
    for iter = 1:iter_max # pseudo transient stepping
      # update damped momentum residuals
      dVxdτ0 .= dVxdτ .+ dampx.*dVxdτ0
      dVydτ0 .= dVydτ .+ dampy.*dVydτ0

      # compute 2nd order invariant of the deviatoric strain rate tensor
      @parallel compute_2invar!(ϵii2, Vx, Vy, dx, dy, nx, ny)

      # compute viscosity at cell centers and vertices
      @parallel compute_viscosity!(ηc, ηv, ϵii2, n, T, T0, θη)  

      # residuals
      dPdτ  = - ∇V
      dVxdτ = diff(τxy[2:end-1,:],dims=2)./dy .+ diff(τxx,dims=1)./dx
      dVydτ = diff(τxy[:,2:end-1],dims=1)./dx .+ diff(τyy,dims=2)./dy
      dTdτ  = (Tt.-T)./dt .- (diff(qx,dims=1)./dx .+ diff(qy,dims=2)./dy) .+ S  
      # update velocities, pressure and temperature
      Vx[2:end-1,:] .= Vx[2:end-1,:] .+ dτVx.*(dVxdτ .+ dampx.*dVxdτ0) 
      Vy[:,2:end-1] .= Vy[:,2:end-1] .+ dτVy.*(dVydτ .+ dampy.*dVydτ0) 
      P             .= P             .+ dτP .* dPdτ
      T             .= T             .+ dτT .* dTdτ
      # check residuals in L2 norm
      if iter % nout == 0 
        R_P  = sqrt(sum(dPdτ .^2))/length(dPdτ )
        R_Vx = sqrt(sum(dVxdτ.^2))/length(dVxdτ)
        R_Vy = sqrt(sum(dVydτ.^2))/length(dVydτ)
        R_T  = sqrt(sum(dTdτ .^2))/length(dTdτ )
        println(">>time = $(round(it*dt, sigdigits=3)), iterations = $(iter), R_P = $(round(R_P,sigdigits=3)), R_Vx = $(round(R_Vx,sigdigits=3)), R_T = $(round(R_T,sigdigits=3))")
        if max(R_P, R_Vx, R_Vy, R_T) < tol break; end
      end
    end # end pseudo transient stepping

    # visualization
    if do_visu && (it % n_vis == 0)
      opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
      display(heatmap(xc, yc, Tt'; opts...))
    end

  end # end physical stepping 
    
  return
end

TM_2D(; do_visu=true)