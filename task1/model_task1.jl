using DiffEqFlux, Flux, OrdinaryDiffEq, LinearAlgebra, Plots, Statistics
import Flux.Tracker: data # For extracting the real values from a tracked array
# For optimizing:
import Flux: throttle               # For only using callback every n seconds
import Base.Iterators: cycle, take  # Useful to iterate over training data
# For loading data:
using JLD2, FileIO


""" u(t) = zoh(ud, dt)
     ZeroOrderHold hold of function ud,
    assuming u(t) = ud[1] 0<=t<dt """
function zoh(ud, dt)
    u(t,x=0) = ud[Int(div(t, dt)+1)]
    return u
end

# Simulation parameters
# These are the same as the data was generated with
global dt = 0.05
global tend = 10.0
global t = 0:dt:tend
global tspan = (0.0,tend)
N = 15
# Discrete time inputs at times t[k]
uds = [2randn(length(t)) for i = 1:N]
# Continuous time versions using Zero-Order-Hold
uins = [zoh(ud, dt) for ud in uds]

# Motor dynamics from control signals
# You do not have to do any changes to this!
function motors(Ω1, Ω2, Ω3, Ω4)
  b = 2.2e-8
  l = 0.046
  d = 1e-9
  ft = b*(Ω1^2 + Ω2^2 + Ω3^2 + Ω4^2)
  τx = b*l*(Ω3^2 - Ω1^2)
  τy = b*l*(Ω4^2 - Ω2^2)
  τz = d*(Ω2^2 + Ω4^2 - Ω1^2 - Ω3^2)
  return ft, τx, τy, τz
end

function quaddu(nn)
     function quaddu_(u, p, t)
        m = 0.027
        g = 9.81
        Ix = p[1]
        Iy = p[2]
        Iz = p[3]
        fwx = 0.0
        fwy = 0.0
        fwz = 0.0
        # u[13:16] should be set by discrete callback
        # Calculate forces from input signals
        ft, τx, τy, τz = motors(u[13], u[14], u[15], u[16])

        # You have to estimate the extra forces someway here
        # -nn(u,v,w)?
        du1  = u[4] + u[6]*cos(u[1])*tan(u[2]) + u[5]*sin(u[1])*tan(u[2])
        du2  = u[5]*cos(u[1]) - u[6]*sin(u[1])
        du3  = u[6]*cos(u[1])/cos(u[2]) + u[5]*sin(u[1])/cos(u[2])
        du4  = u[6]*u[5]*(Iy - Iz)/Ix + τx/Ix
        du5  = u[4]*u[6]*(Iz - Ix)/Iy + τy/Iy
        du6  = u[4]*u[5]*(Ix - Iy)/Iz + τz/Iz
        du7  = u[6]*u[8] - u[5]*u[9] - g*sin(u[2]) + fwx/m
        du8  = u[4]*u[9] - u[6]*u[7] + g*sin(u[1])*cos(u[2]) + fwy/m
        du9  = u[5]*u[7] - u[4]*u[8] + g*cos(u[2])*cos(u[1]) - ft/m + fwz/m
        du10 = u[9]*(sin(u[1])*sin(u[3]) + cos(u[1])*cos(u[3])*sin(u[2])) - u[8]*(cos(u[1])*sin(u[3])-cos(u[3])*sin(u[1])*sin(u[2])) + u[7]*cos(u[3])*cos(u[2])
        du11 = u[8]*(cos(u[1])*cos(u[3]) + sin(u[1])*sin(u[3])*sin(u[2])) - u[9]*(cos(u[3])*sin(u[1])-cos(u[1])*sin(u[3])*sin(u[2])) + u[7]*cos(u[2])*sin(u[3])
        du12 = u[9]*cos(u[1])*cos(u[2]) - u[7]*sin(u[2]) + u[8]*cos(u[2])*sin(u[1])
        # Rotation matrix R: (body to global) above
        # [c2*c3   (c3*s1*s2 - c1*s3)   (s1*s3 + c1*c3*s2);
        #  c2*s3   (c1*c3 + s1*s2*s3)   (c1*s3*s2 - c3*s1)
        #  -s2      c2*s1                c1*c2
        # ]

        ret = Flux.Tracker.collect([du1,du2,du3,du4,du5,du6,
            du7,du8,du9,du10,du11,du12,
            0,0,0,0]) # The control signals are constant until updated by callback
        return ret
    end
    return quaddu_
end

# State feedback controller for quadcopter, defaults to control signal noise 0
# Ref is ref x, y, z
# You do not have to do any changes to this!
function statefeedback(t, state, ref, refnoise=zeros(4))
    ϕ,θ,ψ,p,q,r,u,v,w,x,y,z = state[1:12]
    rx, ry, rz = ref
    Ω0 = 1734.902407525093
    Ωavg = Ω0 + 1000.0*(z - rz) + 600.0*w # Z points downwards (opposite ty Ω)
    dy = v # Approximation if angles are small
    dϕ = p # Approximation if angles are small
    ϕcomp = 0.7*(-70.0*ϕ - 20.0dϕ) + 1.0*( -15.0*(y-ry) - 12.0*dy)# Extra force to increase ψ
    dx = u
    dθ = q # Approximation if angles are small
    θcomp = 0.7*(-70.0*θ - 20.0dθ) + 1.0*( +15.0*(x-rx) + 12.0*dx)# Extra force to increase θ
    dψ = r
    ψcomp = 0.7*(-70.0*ψ - 20.0dψ)

    # Calculate the four control signals, and possibly add noise to them
    Ω1 = Ωavg - ϕcomp - ψcomp + refnoise[1]
    Ω2 = Ωavg - θcomp + ψcomp + refnoise[2]
    Ω3 = Ωavg + ϕcomp - ψcomp + refnoise[3]
    Ω4 = Ωavg + θcomp + ψcomp + refnoise[4]
    return Ω1, Ω2, Ω3, Ω4
end

# Implement zero-order-hold controller as a callback in diff-eq solver
# ucontrol should be (t,state) -> u
# You do not have to do any changes to this!
function callbackcontroller(ucontrol, t)
    function affectcontroller!(integrator, ucontrol)
        u = integrator.u
        u1, u2, u3, u4 = ucontrol(integrator.t, u)
        integrator.u = Flux.Tracker.collect([u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u1,u2,u3,u4])
    end
    condition = (u,ti,integrator) -> (ti ∈ t)
    affect! = integrator -> affectcontroller!(integrator, ucontrol)
    return DiscreteCallback(condition, affect!, save_positions=(false, true))
end

# Flux.jl defaults to Float32 in their networks, this creates a wrapper for generating
# a network witgh Float64, since we are solving differential equations, this extra accuracy
# could be needed.
Dense64(n1, n2, σ=identity) = Dense(n1, n2, σ,
                            initW = (d...) -> Float64.(Flux.glorot_uniform(d...)),
                            initb = (d...) -> zeros(d...))

## neural-network in diff eq
nn = Chain(Dense(3,3))

# Diff-eq definition, with network
contsystem_rd = quaddu(nn)

# Generate a controller that follows a reference, given continuous references
refcontroller(ref) = (t,state) -> statefeedback(t, state, ref(t))
# # generate a controller that follows a reference, given discrete references "rd"
refcont(rd) = refcontroller(zoh(rd, dt))

# Problem for simulation
prob_model(u0, rd, p_guess) = ODEProblem(contsystem_rd, u0, tspan, p_guess, callback=callbackcontroller(refcont(rd), t))

# Predict x, y, z
function predict(u0, rd, p_guess)
    # Using DiffEqFlux ReverseDiff
    println("Solving in function predict(u0, rd, p_guess)..")
    sol = diffeq_rd(p_guess, prob_model(u0, rd, p_guess),  u0=u0, Tsit5(), tstops=t, saveat=Float64[0.0])
    println("Done")
    # Get the x,y,z coordinates only
    reduce(hcat, getindex.(sol.u, (10:12,)))
end

include("data.jl")

u0s, rds, rdns, yreals = data1()
p_guess = param([6,7,11]*10^-6)
predict(u0s[1], rds[1], p_guess)
print(loss_rd(u0s[1],rds[1],yreals[1]))
function loss_rd(u0,rds,y_real)
    pred = predict(u0,rds,p_guess)
    diff = y_real.-pred
    loss = norm([ norm(diff[i]) for i in 1:size(diff)[2]],2)
    return loss
end

traindata = cycle(zip(u0s,rds,yreals))
#Optimizer ADAM, see ?ADAM for info
opt = ADAM()
# Set learning rate without forgetting internal state
opt.eta = 10^-7
# Train over all parameters in nn, as well as p using loss-function loss_rd
# Take the first 300 of (infinite) cycling traindata
# Call the callback function no more than every 2 seconds

cb2 = function ()
    #display(loss_rd(u0s[1],rds[1],yreals[1]))
    plt = plot(layout=2)
    traject = map(Tracker.data,(predict(u0s[1],rds[1],p_guess)[1,:]))
    plot(traject)
    display(plot!(yreals[1][1,:]))

    println(loss_rd(u0s[1],rds[1],yreals[1]))
    #Flux.stop()
end
Flux.train!(loss_rd, params(p_guess), take(traindata,100), opt, cb = throttle(cb2,15))
