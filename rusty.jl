println("lets try it")
#import Pkg
#Pkg.activate("~/Documents/rusty science fair 2019/node")
using Flux, DiffEqFlux, DifferentialEquations, Plots
# initial condition of True Pendulum
u0 = Float32[π*0.5; 0.]  # angle, angular velocity
g_over_l = 9.8
# simulation resolution and time span
datasize = 50
tspan = (0.0,sqrt(g_over_l)*3)

#Define a new kind of layer.
#The whole process of making Flux pay
# attention to parameters is confusing
# it appears the parameter list has to be a tuple
# but that's not enough.  There is something
# to do with @treelike (or @fuctor in the new version)
# that I don't fully get.  But the upshot is that
# if you add parameters in the wrong way it will show
# the parameters in the list of variables to optimize
# but then it won't optimize all the variables in the list
# Fow now till I work this out better then solution is to
# use a cookbook style of creating a layer that does work.
# The good news about this style is that it also makes the
# parameter list retain the names of the layers not just
# the variable types, making it self organizing and easy to
# read.

# Cookebook layer
# create a struct to hold your paramters.
# although Structs are not Mutable they can contain
# a list, which itself is mutable.

struct MultLayer
    mp  # this should be typed to a TrackedRealArray
        # but I have had some problems getting the
        # typing to work right.
end

# Now were going to make an factory/intializer for the struct
# We could make this iniatilizer have some naming convention like
#  MakeStruct  or MakeMultLayer
# A slightly more confusing to code but less confusing to use
# is to take advantage of polymorphism and reuse the
# same name as the struct but with a different type signature


``` x: expects a real array or tuple (I think) of parameters
    dummy: this is a horrible kludge! I needed to give this
            a different signature from the struct and since
                I have been failing to type the stuct it takes
                Any type.  So I needed to add one extra arg
                to this arg list to force a new signature.
                you can uses any argument for dummy and it will
                ignore it
```
function MultLayer(x, dummy)
    MultLayer(Flux.param(x)) # cast the array to a tracked parameter list
                            # then create and return the struct
end


# And Finally this is where you define the custom action
# your layer will perform.
# it gives all structs of type MultLayer a method
# note the method is not a named method, since julia does not
# have any notion of object oriented methods! Instead
# it is called using the name of the struct instance followed
# by a list of parameters with the specified signature
# of the Layer that FLux expects.  And Flux just
# calls the layer with an Anstract array output by the proceeding  layer.
# So if the input is a vector then Flux is going to supply this
# as either a vector or a mini-block of vectors.
# therefore your code needs to handle both a lone 1-D vector
# and a 2-D vector with the other dimension being the miniblock

function (a::MultLayer)(x::AbstractArray)
    a.mp*x # this just multiplies the input by
              # the firts parameter value.  typically this
              # will be a scalar.  But it could also be
              # a matric itself.  If it is a matrix then it
              # has to match the dimension of the input Vector
end


# This is the magic trick that allows Flux to correctly
# parse this parameter.  I think what it does is
# it modifes the Struct name space to give a function
# that know how to parse the struct to extract the parameter list
Flux.@treelike MultLayer


struct myDense2
  W
  b
  beta
  activation::Function
end

function myDense2(W, b, beta, activation,dummy)
    myDense2(Flux.param(W),Flux.param(b),Flux.param(beta),activation::Function )# cast the array to a tracked parameter list
                            # then create and return the struct
end

function (a::myDense2)(x::AbstractArray)
    u=(a.W*x +a.b)
    a.beta.*a.activation.(u)+u  #not quite resnet
end

Flux.@treelike myDense2
#define the true data set


struct TrueModel
  mat
end

function TrueModel(g_over_l::Real=9.8f0)
  TrueModel( [ 0.0f0 1.0f0 ; Float32[-g_over_l] 0.0f0 ] )
end



Truth = TrueModel(9.8f0)

function trueODEfuncComplex(du,u,p,t;TM=Truth)
      du[1] = u[2]
      du[2] = TM.mat[2,1]*sin(u[1])
 end




function trueODEfuncSimple(du,u,p,t; TM = Truth)
     du .= TM.mat*u  #Note! the .= does this in-place modifying the input var
     #true_A = [-0.1 2.0; -2.0 -0.1]
     #du .= ((u.^3)'true_A)'
 end



trueODEfunc0=trueODEfuncComplex

function trueODEfunc1(du,u,p,t; TODE = trueODEfuncComplex)
    TM(du,u,p,t)  #Note! the .= does this in-place modifying the input var
    #true_A = [-0.1 2.0; -2.0 -0.1]
    #du .= ((u.^3)'true_A)'
end
# function trueODEfunc(du,u,p,t; TM = Truth)
#     du .= Truth.mat*u  #Note! the .= does this in-place modifying the input var
#     #true_A = [-0.1 2.0; -2.0 -0.1]
#     #du .= ((u.^3)'true_A)'
# end


# define the time span to run the simulation
t = range(tspan[1],tspan[2],length=datasize)

# Set up this ODE as a problem
prob = ODEProblem(trueODEfuncComplex,u0,tspan)

# solve the ODE for the time series
# save the points at the specified time points.
ode_data = Array(solve(prob,Tsit5(),saveat=t))

# lets see it
plot(ode_data[1,:], label="Angle")
plot!(ode_data[2,:], label="Omega (velocity)")
# Now were going to set up neural ODE which
# is going to learn how to fake the time series



# initial guess
pu = [0.0f0 1.0f0;
     -9.7f0 0.0f0]
# I'm adding in this variation. A single param to Rackaukus's example

# create the neural derivative
#dudt = Chain(x -> x.^3,
# dudt = Chain(MultLayer(pu,"d"),
#              Dense(2,3,tanh),
#              Dense(3,2))

function setupNet1(Layer; p=pu)
  dudt = Chain(x->(p*x),
     myDense2([ 1.0f0 0.0f0 ; 0.0f0 1.0f0 ], [0.0f0, 0.0f0], [0.0f0], tanh,"m"),
     myDense2([ 1.0f0 0.0f0 ; 0.0f0 1.0f0 ], [0.0f0, 0.0f0], [0.0f0], identity,"m")
  ) # Flux won't find params without Chain
  println(dudt)
  ps = Flux.params(dudt)
  println(ps)
  println(Truth.mat)
  return dudt, ps
end

# create the Layer
p1 = MultLayer(pu,"monkey")
println("p1 ",p1.mp)
dudt, ps = setupNet1(MultLayer)
println("Where traing this ", ps)

n_ode = x->neural_ode(dudt,x,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)
pred = n_ode(u0) # Get the prediction using the correct initial condition
#scatter(t,ode_data[1,:],label="data")
#scatter!(t,Flux.data(pred[1,:]),label="prediction")

#loss function

function predict_n_ode(u=u0)
  n_ode(u)
end

loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())

# Now we will train for a certain number of iterations

# define a monitor we can watch
cb1 = function () #callback function to observe training
    display(loss_n_ode())
    display(("ps",ps))
  # plot current prediction against data
    cur_pred = Flux.data(predict_n_ode())
    pl =plot(t,ode_data[1,:],label="data")
    scatter!(pl,t,cur_pred[1,:],label="prediction")
    display(plot(pl))
end

cb1()

function TrainIt(Iterations::Int=3; opt = ADAM(0.1), cb = cb1, ps=ps, loss_n_ode=loss_n_ode)

  # Display the ODE with the initial parameter values.
  println(u0)
  cb()

  data = Iterators.repeated((), Iterations)

  Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
end
TrainIt(100,opt=ADAM(0.001))
TrainIt()
TrainIt()
TrainIt()
TrainIt()
TrainIt()
TrainIt(100)

#
# function trueODEfunc(du,u,p,t)
#     du[1] = u[2]
#     du[2] = -g_over_l*sin(u[1])
# end
# t = range(tspan[1],tspan[2],length=datasize)
# prob = ODEProblem(trueODEfunc,u0,tspan)
# ode_data = Array(solve(prob,Tsit5(),saveat=t))


# this is a control:  it is identical to the model Rauckaukus used.
# It's not a good model for the pendulum but it is a working model

dudt = Chain(x -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))
ps = Flux.params(dudt)

ppp = param(Float32[0.0 1.0 ; -9.0 0.0 ]) # Initial Parameter Vector
dudt = Chain(x ->  ppp*x)  # nothing is actually getting chained !
ps = Flux.params(ppp)

ppp = param(Float32[0.0 1.0 ; -9.0 0.0 ]) # Initial Parameter Vector
dudt = Chain(   x -> (x'ppp')' ) # pointless Chain?

ps = Flux.params(ppp,dudt)

ppp = param(Float32[0.0 1.0 ; -9.0 0.0 ]) # Initial Parameter Vector
dudt =  x -> (x'ppp')'

ps = Flux.params(ppp)

ppp = param(Float32[0.0 1.0 ; -9.0 0.0 ]) # Initial Parameter Vector
dudt = Chain( Dense(2,2) , x -> (x'ppp')' ) # why would adding Dense() to this Chain() matter?

ps = Flux.params(ppp,dudt)

ppp = param(Float32[0.0 1.0 ; -9.0 0.0 ]) # Initial Parameter Vector
dudt = Chain( Dense(2,2) , x -> ppp*x )  #  I removed all the transposes and swapped order of matricies.

ps = Flux.params(ppp,dudt)

function dumb(dims...)
    foo = zeros(Float32,dims...)
    foo[1,2] = 1.0f0
    foo[2,1] = 1.0f0
    return foo
end

ppp = param(Float32[0.0 1.0 ; -9.7 0.0 ]) # Initial Parameter Vector

struct boo
    ppp
end

function boo(x)
    return x -> ppp*x
end


dudt = Chain( Dense(2,2, identity,  initW =  dumb), x-> ppp*x, Dense(2,2, identity,  initW =  dumb)   )  #  I removed all the transposes and swapped order of matricies.

ps = Flux.params(ppp,dudt)

dudt([1 ,2])

dudt(u0)

n_ode = x->neural_ode(dudt,x,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)


pred = n_ode(u0) # Get the prediction using the correct initial condition

#scatter(t,ode_data[1,:],label="data")

#scatter!(t,Flux.data(pred[1,:]),label="prediction")

#loss function
function predict_n_ode()
  n_ode(u0)
end
loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())

data = Iterators.repeated((), 200)  # we just want to see if it will Train-- don't care how well, so 2 steps
opt = ADAM(0.05)
cb = function () #callback function to observe training
  display(loss_n_ode())
  # plot current prediction against data
  cur_pred = Flux.data(predict_n_ode())
  pl = scatter(t,ode_data[1,:],label="data")
  scatter!(pl,t,cur_pred[1,:],label="prediction")
  display(plot(pl))
  println(ps)
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

MethodError: no method matching back!(::Float32)
Closest candidates are:
  back!(::Any, !Matched::Any; once) at /Users/cems/.julia/packages/Tracker/cpxco/src/back.jl:75
  back!(!Matched::Tracker.TrackedReal; once) at /Users/cems/.julia/packages/Tracker/cpxco/src/lib/real.jl:14
  back!(!Matched::TrackedArray) at /Users/cems/.julia/packages/Tracker/cpxco/src/lib/array.jl:68

Stacktrace:
 [1] gradient_(::getfield(Flux.Optimise, Symbol("##15#21")){typeof(loss_n_ode),Tuple{}}, ::Tracker.Params) at /Users/cems/.julia/packages/Tracker/cpxco/src/back.jl:4
 [2] #gradient#24(::Bool, ::typeof(Tracker.gradient), ::Function, ::Tracker.Params) at /Users/cems/.julia/packages/Tracker/cpxco/src/back.jl:164
 [3] gradient at /Users/cems/.julia/packages/Tracker/cpxco/src/back.jl:164 [inlined]
 [4] macro expansion at /Users/cems/.julia/packages/Flux/dkJUV/src/optimise/train.jl:71 [inlined]
 [5] macro expansion at /Users/cems/.julia/packages/Juno/oLB1d/src/progress.jl:134 [inlined]
 [6] #train!#12(::getfield(Main, Symbol("##43#44")), ::typeof(Flux.Optimise.train!), ::Function, ::Tracker.Params, ::Base.Iterators.Take{Base.Iterators.Repeated{Tuple{}}}, ::ADAM) at /Users/cems/.julia/packages/Flux/dkJUV/src/optimise/train.jl:69
 [7] (::getfield(Flux.Optimise, Symbol("#kw##train!")))(::NamedTuple{(:cb,),Tuple{getfield(Main, Symbol("##43#44"))}}, ::typeof(Flux.Optimise.train!), ::Function, ::Tracker.Params, ::Base.Iterators.Take{Base.Iterators.Repeated{Tuple{}}}, ::ADAM) at ./none:0
 [8] top-level scope at In[35]:1

?identity

S

BN

struct reense{F,S,T}
  W::S
  b::T
  σ::F
end

rr = reense(4,3,2)

rr.W

rr.W = 1.0f0

function reense(a::Int64, b::Int64, c::Int64)
    println("ouch")
end


reense(4,3,2)

rx = reense(4,3.0,2)

rt

rt.W


'''
  failed
    Status `~/.julia/environments/v1.3/Project.toml`
  [aae7a2af] DiffEqFlux v0.7.0
  [0c46a032] DifferentialEquations v6.9.0
  [7073ff75] IJulia v1.20.2
  [91a5bcdd] Plots v0.28.4
trying
    Status `~/.julia/environments/v1.3/Project.toml`
  [aae7a2af] DiffEqFlux v0.4.0
  [0c46a032] DifferentialEquations v6.6.0
  [587475ba] Flux v0.10.0
  [7073ff75] IJulia v1.20.2
  [91a5bcdd] Plots v0.28.4

# https://github.com/FluxML/Flux.jl/issues/713
# https://github.com/FluxML/model-zoo/tree/master/contrib/diffeq
# https://github.com/FluxML/Flux.jl/issues/198

struct Affine
  W
  b
end


Affine(in::Integer, out::Integer) =
  Affine(param(randn(out, in)), param(randn(out)))

(m::Affine)(x) = m.W * x .+ m.b

Flux.treelike(Affine)

m = Affine(2,3)

m.W
# output
Tracked 3×2 Array{Float64,2}:
  0.213468   1.91803
  1.29768   -0.0585969
 -0.414205   1.11279

params(m)

'''
