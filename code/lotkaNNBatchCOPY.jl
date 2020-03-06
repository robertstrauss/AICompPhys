using Flux, DiffEqFlux, DifferentialEquations, LinearAlgebra
import Plots

include("MyUtils.jl")





# truth ODE

## 2-animal lotka-volterra
u0 = gc(Float32[0.44249; 4.6280594])
A = Float32[0.0 -0.9; 0.5 0.0]
b = Float32[1.3, -1.8]
dudt = lv(A, b)

## use all channels
cmap1 = Float32[1.0]
cmap2 = Float32[1.0]

"""
## 3-animal lotka-volterra
u0 = gc(Float32[ 2.5; 5.0; 7.5 ])
A = Float32[ 0.0   -0.535  0.532 ;
             0.531  0.0   -0.536 ;
            -0.534  0.533  0.0   ]
b = Float32[ 0.4, -0.2, -0.2 ]
dudt = lv(A,b)

## only train on two of three channels
cmap1 = [1.0,1.0,0.0]
cmap2 = [1.0,1.0,0.0]
"""




# time span
tspan = (0.0f0, 3.0f0)
t = range(tspan..., length = 100)

# recalculate truth and plot it against prediction
truth = neural_ode(dudt, u0, tspan, saveat = t)
Plots.plot(t, truth')


# empty neural net of swishes
## 2-animal net
dudt_train = Chain(
    Dense(2, 32, swish),
    Dense(32, 32, swish),
    Dense(32, 32, swish),
    Dense(32, 2),
)
## tracking, necessary to train
tracking = Flux.params(dudt_train)

# neural net solver
n_ode(u) = neural_ode(dudt_train, u, tspan, Tsit5(), saveat = t) #,reltol=1e-7,abstol=1e-9)


# single case
struct datum
    n_ode::Function #NeuralODE
    labels::AbstractArray
    u0::AbstractArray
    t
end

# for parallel cases in a batch
struct batch
    n_ode::Function
    labels::AbstractArray
    u::AbstractArray
    t
end


## loss functions

function loss(; n_ode = n_ode, truth = truth, u0 = u0)
    sum(abs2, (n_ode(u0) .* cmap1 - truth .* cmap2))
end

loss(d::datum) = loss(n_ode=d.n_ode,truth=d.labels,u0=d.u0)

loss(b::batch) = loss(n_ode=b.n_ode,truth=b.labels,u0=b.u)

# do loss but dont throw away prediction
function losspred(;n_ode = n_ode,truth = truth,u0=u0)
    a = Flux.data(n_ode(u0))
    loss = sum(abs2, (a .* cmap1 - truth .* cmap2))#.*[1.0f0 1.0f0 0.0f0]' )
    return loss, a
end

losspred(d::datum) = losspred(n_ode=d.n_ode,truth=d.labels,u0=d.u0)

losspred(b::batch) = losspred(n_ode=b.n_ode,truth=b.labels,u0=b.u)





## callbacks for output to plot prediction and truth
cb = function(;truth=truth, losspred=losspred, t=t, pl=nothing)
    loss, cur_pred = losspred()
    # println(truth)
    pl1 = (pl==nothing) ? Plots.plot() : pl
    Plots.plot!(pl1, t, truth', label = "truth")
    Plots.scatter!(pl1, t, cur_pred', label = "prediction")
    if (pl==nothing)
        display(pl1)
    end
    display(loss)
end

cb(d::datum) = cb(truth=d.labels,losspred=()->losspred(d),t=d.t)

cb = function(b::batch)
    pl = Plots.plot()
    for i in 1:1 # only plot first five cases in batch b
        cb(
            truth=b.labels[:,i,:],
            losspred=()->losspred(n_ode=b.n_ode,truth=b.labels[:,i,:],u0=b.u[:,i]),
            t=b.t,
            pl=pl
        )
    end
    display(pl)
end

function timeDatum(startTime::Float32, stopTime::Float32; u0 = u0)
    tspan = (startTime, stopTime)
    t = range(tspan..., length = 30)
    truth = neural_ode(dudt, u0, (0, stopTime))
    u1 = truth(startTime)
    n_ode2(u) = neural_ode(dudt_train, u, tspan, Tsit5(), saveat = t)
    truth2 = neural_ode(dudt, u1, tspan)
    labels = Flux.data(truth2(t))
    datum(n_ode2, labels, u1, t)
end

function uDatum(u0, dur)
    tspan = (0.0f0, dur)
    t = range(tspan..., length = 30)
    u1 = u0
    n_ode2(u) = neural_ode(dudt_train, u, tspan, Tsit5(), saveat = t)
    truth2 = neural_ode(dudt, u1, tspan)
    labels = Flux.data(truth2(t))
    datum(n_ode2, labels, u1, t)
end

function uBatch(u0s, dur)
    tspan = (0.0f0, dur)
    t = range(tspan..., length = 30)
    n_ode(u) = neural_ode(dudt_train, u, tspan, Tsit5(), saveat = t)
    truth = neural_ode(dudt, u0s, tspan)
    labels = Flux.data(truth(t))
    batch(n_ode, labels, u0s, t)
end

dataInits = [
        (uDatum([i * 1.0f0, 3.5f0 + 3.0f0 * sin(i * 1.0f0)], 3.0f0),)
    for i in range(0.0f0,6.0f0,length = 8)]


inits(a) = hcat([
        [i + a, 3.5f0 + 3.0f0 * sin(i) + a]
    for i in range(0.0f0,6.0f0,length = 8)]...)
dataBatch = [
        (uBatch(inits(a),4.0f0,),)
    for a in range(0.1f0, 1.0f0, length = 10)]


@Flux.epochs 150 Flux.train!(
    loss,
    tracking,
    dataBatch,
    ADAM(1.0E-3);
    cb = Flux.throttle(()->cb(dataBatch[1]...), 3),
)
cb(dataBatch[1]...)
Flux.train!(
    loss,
    tracking,
    Iterators.repeated((), 500),
    Descent(1.0E-6);
    cb = Flux.throttle(cb, 3),
)
