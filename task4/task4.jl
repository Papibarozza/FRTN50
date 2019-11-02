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
         # Rotation matrix R (3x3): (body to global)
         c1 = cos(data(u[1]))
         c2 = cos(data(u[2]))
         c3 = cos(data(u[3]))
         s1 = sin(data(u[1]))
         s2 = sin(data(u[2]))
         s3 = sin(data(u[3]))

         R = [[c2*c3   (c3*s1*s2 - c1*s3)   (s1*s3 + c1*c3*s2)];
             [c2*s3   (c1*c3 + s1*s2*s3)   (c1*s3*s2 - c3*s1)];
             [-s2      c2*s1                c1*c2]]


        m = 0.027
        g = 9.81
        Ix = p[1]
        Iy = p[2]
        Iz = p[3]

        wind = R'*p[10:12]*10^4 #Transform from inertial frame to body frame
        #println(wind)
        #println(wind)
        #println(-10^5*(p[4]*u[7]+p[7]*abs(u[7])*u[7]))
        wind_x = wind[1]
        wind_y = wind[2]
        wind_z = wind[3]


        fwx = -10^5*(p[4]*u[7]+p[7]*abs(u[7])*u[7])+wind_x
        fwy = -10^5*(p[5]*u[8]+p[8]*abs(u[8])*u[8])+wind_y
        fwz = -10^5*(p[6]*u[9]+p[9]*abs(u[9])*u[9])+wind_z
        #println([Ix,Iy,Iz])
        #println([fwx,fwy,fwz])
        # u[13:16] should be set by discrete callback
        # Calculate forces from input signals
        ft, τx, τy, τz = motors(u[13], u[14], u[15], u[16])
        #println(u)

        # You have to estimate the extra forces someway here

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
nn = Chain(Dense64(3,3),Dense64(3,3))

# Diff-eq definition, with network
contsystem_rd = quaddu(nn)

# Generate a controller that follows a reference, given continuous references
refcontroller(ref,ref_rdn) = (t,state) -> statefeedback(t, state, ref(t), ref_rdn(t))
# # generate a controller that follows a reference, given discrete references "rd"
refcont(rd,rdns) = refcontroller(zoh(rd, dt),zoh(rdns,dt))

# Problem for simulation
prob_model(u0, rd,rdns,p_guess) = ODEProblem(contsystem_rd, u0, tspan, p_guess, callback=callbackcontroller(refcont(rd,rdns), t))

# Predict x, y, z
function predict(u0, rd,rdns, p_guess)
    # Using DiffEqFlux ReverseDiff
    sol = diffeq_rd(p_guess, prob_model(u0, rd,rdns, p_guess),  u0=u0, Tsit5(), tstops=t, saveat=Float64[0.0])
    # Get the x,y,z coordinates only
    reduce(hcat, getindex.(sol.u, (10:12,)))
end

#include("data.jl")

#u0s, rds, rdns, yreals = data4()
#p_guess = param([6,7,11,5,5,5,5,5,5,5,5,5]*10^-6)
#predict(u0s[1], rds[1],rdns[1],p_guess)
#print(loss_rd(u0s[1],rds[1],rdns[1],yreals[1]))
function loss_rd2(u0,rds,rdns,y_real)
    pred = predict(u0,rds,rdns,p_guess)
    diff = y_real-pred
    #loss = norm([ norm(diff[i]) for i in 1:size(diff)[2]],2)+norm(p_guess,2)
    loss = norm([ norm(diff[i]) for i in 1:size(diff)[2]],2)
    return loss
end

#traindata = cycle(zip(u0s,rds,rdns,yreals))
#Optimizer ADAM, see ?ADAM for info
opt = ADAM()
# Set learning rate without forgetting internal state
opt.eta = 10^-7
# Train over all parameters in nn, as well as p using loss-function loss_rd
# Take the first 300 of (infinite) cycling traindata
# Call the callback function no more than every 2 seconds
function plotfun(i=1)
    plt = plot(layout=2)
    trajectx = map(Tracker.data,(predict(u0s[i],rds[i],rdns[i],p_guess)[1,:]))
    trajecty = map(Tracker.data,(predict(u0s[i],rds[i],rdns[i],p_guess)[2,:]))
    trajectz = map(Tracker.data,(predict(u0s[i],rds[i],rdns[i],p_guess)[3,:]))
    display(plot(trajectx,lab="prediction-x"))
    display(plot!(trajecty,lab="prediction-y"))
    display(plot!(trajectz,lab="prediction-z"))
    display(plot!(yreals[i][1,:],lab="real-x"))
    display(plot!(yreals[i][2,:],lab="real-y"))
    display(plot!(yreals[i][3,:],lab="real-z"))
    println(loss_rd2(u0s[i],rds[i],rdns[i],yreals[i]))
end
cb2 = function ()
    #display(loss_rd(u0s[1],rds[1],yreals[1]))
    #plt = plot(layout=2)
    #traject = map(Tracker.data,(predict(u0s[1],rds[1],rdns[1],p_guess)[1,:]))
    #plot(traject,lab="prediction")
    #display(plot!(yreals[1][1,:],lab="real"))
    println(loss_rd2(u0s[1],rds[1],rdns[1],yreals[1]))
end
#println(take(traindata,1))
#Flux.train!(loss_rd2, params(p_guess), take(traindata,200), opt, cb = throttle(plotfun,10))
include("data.jl")
include("../validation_tools.jl")
u0s, rds, rdns, yreals = data4()
#p_guess = param([6,7,11,5,5,5,5,5,5,5,5,5]*10^-6)
p_guess = param([7.19377e-6,  5.79928e-6,  1.77358e-5,  7.38943e-8,  2.92086e-7,  5.09897e-7,  1.10199e-7,  1.6916e-7,  7.5907e-7,  5.18762e-7,  7.73919e-7,  -1.29269e-6])

#Flux.train!(loss_rd2, params(p_guess), take(traindata,300), opt, cb = throttle(plotfun,5))
params_guessed = zeros(12,5)
loss = zeros(1,25)
i=1
for k = 1:5
    println(k)
    idxs_test, idxs_train = partition_idxs(15,5)
    global p_guess = param([7.19377e-6,  5.79928e-6,  1.77358e-5,  7.38943e-8,  2.92086e-7,  5.09897e-7,  1.10199e-7,  1.6916e-7,  7.5907e-7,  5.18762e-7,  7.73919e-7,  -1.29269e-6])
    traindata = cycle(zip(u0s[idxs_train],rds[idxs_train],rdns[idxs_train],yreals[idxs_train]))
    testdata = cycle(zip(u0s[idxs_test],rds[idxs_test],rdns[idxs_test],yreals[idxs_test]))
    Flux.train!(loss_rd2, params(p_guess), take(traindata,300), opt,cb=cb2)
    for j in idxs_test
        global loss[i]=data(loss_rd2(u0s[j],rds[j],rdns[j],yreals[j]))
        global i = i+1
    end
    params_guessed[:,k] = data(p_guess)
end
println(loss/50)
println(params_guessed*10^6)
