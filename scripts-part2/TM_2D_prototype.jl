# Part 2: Thermomechanical coupling
using Plots, Printf

@views function TM_2D_prototype(; do_visu = false)
  # Physics
  Lx, Ly = 0.86038, 0.86038     # domain size 
  n      = 3                    # power law rheology exponent
  Vbc    = 66.4437              # boundary velocity  
  T0     = 49.3269 / n          # initial temperature
  r      = 0.0737               # perturbation radius
  Tamp   = 0.1*T0               # perturbation temperature

  # Numerics
  nx, ny = 32*2^2-2, 32*2^2-2       # grid size
  dx, dy = Lx/nx, Ly/ny         # grid step
  tol    = 1e-5                 # non-linear tolerance
  ξ      = 10.0                 # physical steps reduction for temperature
  θp     = 0.5                  # PT steps reduction for pressure 
  θv     = 0.5                  # PT steps reduction for velocity
  θT     = 0.5                  # PT steps reduction for temperature
  ηb     = 1.0                  # numerical bulk viscosity
  θη     = 0.9                  # relaxation factor for viscosity
  ν      = 4.0                  # damping factor for velocity residuals
  nt     = 20                   # number of time steps
  iter_max = 1e6                # max PT iterations
  nout   = 100                  # residuals checking step
  n_vis  = 1                    # visualization step

  # Derived numerics
  dt    = ξ*min(dx, dy)^2/4.1   # physical time step  
  dampx = 1 - ν/nx              # damping for x-momentum residuals
  dampy = 1 - ν/ny              # damping for y-momentum residuals
    
  # Generate staggered grid
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
  qx     = zeros(Float64, nx+1, ny  )   # x-heat flux 
  qy     = zeros(Float64, nx  , ny+1)   # y-heat flux 
  ϵii2   = zeros(Float64, nx  , ny  )   # 2nd order invariant
  S      = zeros(Float64, nx  , ny  )   # source term
  ∇V     = zeros(Float64, nx  , ny  )   # velocity divergence
  ϵxx    = zeros(Float64, nx  , ny  )   # xx-strain rate 
  ϵyy    = zeros(Float64, nx  , ny  )   # yy-strain rate   
  ϵxyv   = zeros(Float64, nx+1, ny+1)   # xy-strain rate (defined at vertices)
  τxx    = zeros(Float64, nx  , ny  )   # xx-normal stress
  τyy    = zeros(Float64, nx  , ny  )   # yy-normal stress
  τxy    = zeros(Float64, nx+1, ny+1)   # xy-shear stress
  dVxdτ  = zeros(Float64, nx-1, ny  )   # x-momentum residuals
  dVydτ  = zeros(Float64, nx  , ny-1)   # y-momentum residuals
  dPdτ   = copy(P)                      # continuity residuals
  dTdτ   = copy(T)                      # temperature residuals
  dVxdτ0 = copy(dVxdτ)
  dVydτ0 = copy(dVydτ)

  # Initial conditions
  Vx  .=  Vbc.*X_fx./Lx
  Vy  .= -Vbc.*Y_fy./Ly
  T[(Xc.^2 .+ Yc.^2) .< r^2] .= Tamp

  # Preparation of visualisation
  if do_visu
    ENV["GKSwstype"]="nul"; if isdir("TM2D_out")==false mkdir("TM2D_out") end; loadpath = "./TM2D_out/"; anim = Animation(loadpath,String[]); println("Animation directory: $(anim.dir)")
  end

  for it = 1:nt # Physical stepping
    Tt    .= T            # temperature updates from previous timestep
    for iter = 1:iter_max # pseudo transient stepping
      # update damped momentum residuals
      dVxdτ0 .= dVxdτ .+ dampx.*dVxdτ0
      dVydτ0 .= dVydτ .+ dampy.*dVydτ0
      # strain rates
      Vxbc = [ Vx[:,1]   Vx  Vx[:,end] ]  # extended with free slip boundaries                      
      Vybc = [ Vy[1,:]'; Vy; Vy[end,:]']  # extended with free slip boundaries
      ∇V   = diff(Vx,dims=1)./dx .+ diff(Vy,dims=2)./dy
      ϵxx  = diff(Vx,dims=1)./dx .- 1/2 .* ∇V
      ϵyy  = diff(Vy,dims=2)./dy .- 1/2 .* ∇V
      ϵxyv = 0.5.*(diff(Vxbc,dims=2)./dy .+ diff(Vybc,dims=1)./dx)
      ϵxyc = 0.25.*(ϵxyv[1:end-1,1:end-1] .+ ϵxyv[2:end,1:end-1] .+ ϵxyv[1:end-1,2:end] .+ ϵxyv[2:end,2:end]) # interpolate
      ϵii2 = 0.5.*(ϵxx.^2 .+ ϵyy.^2) .+ ϵxyc.^2
      # viscosity
      η0                   = ϵii2.^(-(1-1/n)/2).*exp.( -T.*(1 ./(1 .+ T./T0)) )   # physical viscosity
      ηc                  .= exp.(θη.*log.(ηc) + (1-θη).*log.(η0))                # effective viscosity
      ηv[2:end-1,2:end-1] .= 0.25.*(ηc[1:end-1,1:end-1] .+ ηc[2:end,2:end] .+ ηc[1:end-1,2:end] .+ ηc[2:end,1:end-1]) # interpolate  
      # stress 
      τxx = 2 .* ηc .*(ϵxx .+ ηb.*∇V) .- P  
      τyy = 2 .* ηc .*(ϵyy .+ ηb.*∇V) .- P  
      τxy = 2 .* ηv .* ϵxyv                 
      # heat fluxes and source
      qx[2:end-1,:] = -diff(T,dims=1)./dx
      qy[:,2:end-1] = -diff(T,dims=2)./dy
      S             = 4 .* ηc .* ϵii2       
      # pseudo-time steps
      dτP  = θp*4.1/ min(nx,ny)   .* ηc .* (1.0+ηb)
      dτVx = θv/4.1*(min(dx,dy)^2 ./( 0.5.*(ηc[2:end,:] + ηc[1:end-1,:]) ))./(1+ηb)
      dτVy = θv/4.1*(min(dx,dy)^2 ./( 0.5.*(ηc[:,2:end] + ηc[:,1:end-1]) ))./(1+ηb)
      dτT  = θT/4.1* min(dx,dy)^2
      # residuals
      dVxdτ = diff(τxy[2:end-1,:],dims=2)./dy .+ diff(τxx,dims=1)./dx
      dVydτ = diff(τxy[:,2:end-1],dims=1)./dx .+ diff(τyy,dims=2)./dy
      dPdτ  .= - ∇V
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
      opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
      display(heatmap(xc, yc, T'; opts...)); frame(anim)
    end

  end # end physical stepping 

  # Save animation
  if (do_visu ) gif(anim, "TM_2D.gif", fps = 5)  end
    
  # return dVxdτ0, dVydτ0, ηc, ηv, dPdτ, dVxdτ, dVydτ, Vx, Vy, P, T
  return
end

TM_2D_prototype(; do_visu = true)