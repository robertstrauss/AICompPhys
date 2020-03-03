using DifferentialEquations, Flux, Plots, DiffEqFlux, LinearAlgebra


# Here we will do an extremely simple example to demonstrate what Neural ODEs can do

# first, we create a lotka-volterra differential equation and solve it to get the populations of rabbits and wolves over time.
# The solution will be used for truth data to train a neural net acting as a differential equation. This is the neural ODE.
# No real life data is used. "Truth" data is data from the known equations.
# The lotka volterra equations are just being used as an example, to demonstrate neural ODEs.


# First we define some basic globals

# The timespan over which the simulation will occur. Could be in years, decades, or anything
tspan = (0.0f0,16.0f0)
t = range(tspan[1], tspan[2], length=100)


# Lotka-Volterra data structure
struct lv
    A
    b
end

# every instance of lv also doubles as a function, which returns the rate of change
#   of an input population based on the parameters of the lv instance ( b and A )
(m::lv)(x) = x.*( m.b .+ m.A*x)

# this allows flux to look inside the structure to attach trackers to parameters
@Flux.treelike lv


# u represents the state of the system, being a vector of the two populations - rabbits and wolves.
# u0 is the initial condition. These could be in units of millions of individuals, or anything.
u0 = Array(Float32[0.44249; 4.6280594])

# A and b define the constant coeffecient parameters of the system
# they represent the "hungriness" of the wolves and the natural population growth/death rate
A = Float32[0.0 -0.9; 0.5 0.0]
b = Float32[1.3, -1.8]

# the differential equation, a lotka-volterra system using the parametrs given
# dudt is a function that takes in a vector of two populations
# and outputs the rate of change. A and b are the constant paramers of this system
dudt = lv(A, b)

# solution to the equation from the intial condition over the timespan
sol = neural_ode(dudt, u0, tspan, saveat=t)
data = Array(Flux.data(sol)) # strips the autograd trackers from it, to get just an array of data

plot(t, data') # plot the solution over time

# a data structure containing ac and bd, parameters for a lotka-volterra system
# flip - an inverse identity matrix to horizontally reverse data
struct LV
    ac
    bd
    flip
end

# defining a constructor for an LV (lotka volterra) system from parameters
LV(ac, bd) = LV(Flux.param(ac), Flux.param(bd),
                Matrix{Float32}(I, size(ac)[1], size(ac)[1])[:,end:-1:1])

# every instance of and LV structure is also a function
(m::LV)(x) = (m.ac .+ m.bd.*(m.flip*x)).*x

# allows autograd to understand our LV structure and automatically put trackers on its data
Flux.@treelike LV

# the neural net - input layer is 2 wide, three hidden layers are 32 wide, and an output layer is 2 wide.
# swish is used as an activation function on each layer.
dudt_train = Chain(
    Dense(2, 32, swish),
    Dense(32, 32, swish),
    Dense(32, 32, swish),
    Dense(32, 2),
)
# add autograd trackers to everything (all weights and biases) in dudt_train, the neural net
ps = Flux.params(dudt_train)

# function solving (integrating) the neural net as a differential equation for any given initial condition as an input
n_ode = x->neural_ode(dudt_train,x,tspan,Tsit5(),saveat=t)#,reltol=1e-7,abstol=1e-9)

# first prediction - probably utter nonsense, the net hasn't been trained
pred = n_ode(u0)

# wrapper of n_ode function
predict_n_ode(u=u0) = n_ode(u)

# loss function - sums the absolute value squared difference of the solution from the network and the truth data from the actual equations
loss_n_ode() = sum(abs2, data .- predict_n_ode())

cb = function () # callback function to visualize training
    # plots the current solution from the neural net against the truth data

    display(loss_n_ode()) # print out the loss
    # plot current prediction against data
    cur_pred = Flux.data(predict_n_ode()) # solve the neural net as it is currently
    pl = Plots.plot() # create a plot
    t2 = range(tspan[1], tspan[2], length=10) # a timespan over the same range as t but with a specific resolution
    plot!(pl,t,data',label="data") # plot the truth data (apostrophe transposes)
    scatter!(pl,t,cur_pred',label="prediction") # prot the the solution from the neural net
    display(plot(pl)) # show the plot
end

cb() # visualize everything once before training

# function to train the net for iters iteration
function train(iters::Int = 3; opt = ADAM(0.1), callback = cb, ps = ps, loss_n_ode = loss_n_ode)
    callback() # visualize before training
    d = Iterators.repeated((), iters) # empty "batch" - essentially just a counter for the trainer to iterate over
    Flux.train!(loss_n_ode, ps, d, opt, cb = Flux.throttle(callback, 3)) # train the neural net!
    # Flux.throttle(callback, s) slows callback to only be called once ever s seconds
end

# do it! this will train the network for 300 iterations with optimization of ADAM(1E-3). It will be visualized every 3 seconds
train(300,opt=ADAM(1E-3)) # this may need to be run multiple times to get it to match the truth data
# the ADAM value (1E-3) is low because the truth data is periodic. (and for stability)
# if it tries to optimize too quickly, it will just become the median
