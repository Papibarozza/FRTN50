using Pkg
Pkg.add("DiffEqFlux")
Pkg.add("Flux")
Pkg.add("OrdinaryDiffEq")
Pkg.add("LinearAlgebra")
Pkg.add("Plots")
Pkg.add("Statistics")
Pkg.add("JLD2")
Pkg.add("FileIO")
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
    u(t) = ud[Int(div(t, dt)+1)]
    return u
end

N = 15
dt = 0.1
tend = 10.0
t = 0:dt:tend
tspan = (0.0,tend)
# Discrete time inputs at times t[k]
uds = [2randn(length(t)) for i = 1:N]
# Continuous time versions using Zero-Order-Hold
uins = [zoh(ud, dt) for ud in uds]

# Some random initial points
u0s = [param([1.8*rand()-0.9, 0.1*rand(), 0.1*rand(),0]) for i = 1:N]

# Flux.jl defaults to Float32 in their networks, this creates a wrapper for generating
# a network witgh Float64, since we are solving differential equations, this extra accuracy
# could be needed. The same can be done with other types of layers you might use
Dense64(n1, n2, σ=identity) = Dense(n1, n2, σ,
                            initW = (d...) -> Float64.(Flux.glorot_uniform(d...)),
                            initb = (d...) -> zeros(d...))

# Neural network to approximate uncertanty
nn = Chain(Dense64(1,10,relu), Dense64(10,10, relu), Dense64(10,1))
# Real uncertanty, we make sure output is a "TrackedArray"
# A TrackedArray is a special array that can store gradients
nn_real = u -> Flux.Tracker.collect([u[1]^3])

# Out guess of parameters
# "param" takes vector and makes the contents "parameters",
# so that gradients will be tracked with respect to these
p = param([1.1,0.7])

# We use this to simulate the real reference trajectory
# Make params to the types are the same as for p
p_real = param([1.0,1.0])

# Wrapper that given uin and nn returns the function that computes derivative
# This avoids that dudtreal_ refers to global variables nn and uin
function dudt(uin, nn)
    # Return the derivative du/dt given current state u, parameters p, and time t
    function dudtreal_(u::TrackedArray,p,t)
        #Calculate output of neural network, vector as input
        # Type assurtion to make sure we are doing correct and speed up a bit
        nnout = nn(u[1:1])::TrackedArray{Float64,1,Array{Float64,1}}
        # "Flux.Tracker.collect" takes a vector of TrackedReal and turns
        # it into a TrackedArray, which is expected by the optimizer
        # The states of the DiffEq solvers are called u by convention
        du = Flux.Tracker.collect(
                [-1.2*u[1] + 0.1*u[2] + nnout[1],
                -u[2] + u[3] + 0.7*p[1],
                3*p[2]*uin(t) - 0.4*u[3],
                0.0])
        # Last input has derivative 0, and is set by callback in solver and
        # constant in between time samples
        return du
    end
    return dudtreal_
end

# This function will at time points t[k] set the fourth state equal to the first
function callback_setter(t)
    # Condition for update, integrator is internal DiffEq state
    condition = (u,ti,integrator) -> (ti ∈ t)
    # When condition true, make this change to internal state
    function affect!(integrator)
        u = integrator.u
        # Set internal state u
        # We again make sure that output is a TrackedArray
        integrator.u = Flux.Tracker.collect([u[1],u[2],u[3],integrator.u[1]])
    end
    return DiscreteCallback(condition, affect!, save_positions=(false, true))
end

# DifferentialEquations.jl problem. dudt(uin,nn) if the function that calculates derivatives,
# u0 is initial state, tspan is the span of time we solve over,
# p is set of parameters sent to "dudt(uin,nn)", callback is defined as above
prob(uin, u0) = ODEProblem(dudt(uin,nn), u0, tspan, p, callback = callback_setter(t))
# With the real uncertanty, to generate reference data
prob_real(uin, u0) = ODEProblem(dudt(uin,nn_real),u0,tspan,p_real, callback = callback_setter(t))


# Solve example:
# diffeq_rd takes a ODEProblem (prob_real), and solves it, while enabling back-propagation
# to parameters p_real, using solver Tsit5 (see FiffEqFlux.jl and DifferentialEquations.jl)
# inital point is u0s[1], it makes sure to stop at every time in t, so that callback can be triggered,
# saveat makes sure we save result only at tstops
diffeq_rd(p_real, prob_real(uins[1],u0s[1]), Tsit5(), u0=u0s[1], tstops=t, saveat=Float64[0.0])

# Collect the solutions for each input and initial state
# zip iterates over each index in uins and u0s
sols_real = [diffeq_rd(p_real,prob_real(uin,u0),Tsit5(),u0=u0,tstops=t,saveat=Float64[0.0]) for (uin,u0) in zip(uins,u0s)]
# Extract the first state from all solutions (this is what we want to approximate)
# The function "data", takes a TrackedVector (backpropagation vector)
# and returns normal data (without tracked gradients)
yreals = [getindex.(data.(sol_real.u),1) for sol_real in sols_real]

# Predict the FIRST state given uin an u0
function predict_rd(uin, u0)
    # Make sure to stop at each t[k] in t, sp callback is called, saveat makes sure we save result only at tstops
    sol = diffeq_rd(p, prob(uin, u0), Tsit5(), u0=u0, tstops=t, saveat=Float64[0.0])
    # sol.u are all the the states, one vector per time index, get the first in each of them
    Flux.Tracker.collect(getindex.(sol.u,1))
end



# The data we want to train on, as an iterator, zip makes sure that we get
# (uins[1],u0s[1],yreals[1]), (uins[2],u0s[2],yreals[2]), and so on.
# cycle means iterate over all of them an infinite number of times
traindata = cycle(zip(uins,u0s,yreals))

# Loss function, sum of absolute values of difference of real and simulated
function loss_rd(uin, u0, yreal)
    l = sum(abs,yi-yireal for (yi,yireal) in zip(predict_rd(uin, u0), yreal))
    # print loss to see progress later
    display(l)
    return l
end
# Example, returns a tracked value
loss_rd(uins[1], u0s[1], yreals[1])

# Fancy callback function to track progress
cb = function ()
    # Calculate loss of first trajectory in data set
    display(loss_rd(uins[1], u0s[1], yreals[1]))
    plt = plot(layout=2)
    # Plot predicted and real trajectory for first data point
    plot!(plt, data(predict_rd(uins[1], u0s[1]))[:], lab="", c=:red, subplot=1)
    plot!(plt, yreals[1], lab="", c=:blue, subplot=1)
    # Plot approximation of uncertanty
    plot!(plt, -1:0.1:1, reduce(vcat, [data(nn([x])[1]) x^3] for x in -1:0.1:1.0), lab=["nn" "x^3"], subplot=2)
    println(data(p))
    display(plt)
end

# Display progress
cb()

#Optimizer ADAM, see ?ADAM for info
opt = ADAM()
# Set learning rate without forgetting internal state
opt.eta = 0.01
# Train over all parameters in nn, as well as p using loss-function loss_rd
# Take the first 300 of (infinite) cycling traindata
# Call the callback function no more than every 2 seconds
Flux.train!(loss_rd, params(nn,p), take(traindata,300), opt, cb = throttle(cb, 2))

# Compare extimated and real parameters
[p p_real]
