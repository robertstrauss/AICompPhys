using Flux, DiffEqFlux, DifferentialEquations, LinearAlgebra
import Plots, Random

include("MyUtils.jl")

## 3-animal lotka-volterra
# u0 = gc(Float32[ 2.5; 5.0; 7.5 ])
# A = Float32[ 0.0   -0.535  0.532 ;
#              0.531  0.0   -0.536 ;
#             -0.534  0.533  0.0   ]
# b = Float32[ 0.4, -0.2, -0.2 ]
u0 = Float32[ 5.5; 0.6; 15.7]
A = Float32[ 0.0 -0.9  -0.1;
              0.5  0.0  -0.1 ;
              0.3  0.98  0.0]
b = Float32[ 1.3, -1.8*0.5, -2.0] # netter with bears
dudt = lv(A,b)

# blind to a channel
cmap1 = [1.0f0; 1.0f0; 0.0f0]
cmap2 = [1.0f0; 1.0f0; 0.0f0]

# time span
tspan = (0.0f0, 20.0f0)
t = range(tspan..., length = 100)

# recalculate truth and plot it against prediction
truth = neural_ode(dudt, u0, tspan, saveat = t)
Plots.plot(t, truth')


# empty neural net of swishes for predicting initial condition
pred_bears = Chain(
    Dense(12, 32, swish),
    Dense(32, 32, swish),
    Dense(32, 1),
)

# empty neural net of swishes for solving ODE
dudt_train = Chain(
    Dense(3, 32, swish),
    Dense(32, 32, swish),
    Dense(32, 32, swish),
    Dense(32, 3),
)
## tracking, necessary to train
tracking = Flux.params(dudt_train, pred_bears)


function index(ar, i)
    thin =  ar[i...]
    dropdims(thin; dims=tuple(findall(size(thin).==1)...))
end


# neural net solver
function n_ode(u;t=t,tspan=tspan)
    data = cat(neural_ode(dudt, u, (0.0f0,0.5f0), saveat=0.0f0:0.1f0:0.5f0).u...,  dims=3) # only get 2 channels
    bears = pred_bears(hcat(data[1,:,:][:,:],data[2,:,:][:,:])') # predict third channel
    uu = vcat(u[1:2,:,:][:,:], bears) # input vector for neural ode
    index(neural_ode(dudt_train, uu, tspan, Tsit5(), saveat = t),(:,:,:)) # return output
end


# single case
struct datum
    n_ode::Function
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
function cb(;truth=truth, losspred=losspred, t=t, pl=nothing)
    loss, cur_pred = losspred()
    pl1 = (pl==nothing) ? Plots.plot() : pl
    Plots.scatter!(pl1, cur_pred', label = "prediction")
    Plots.plot!(pl1, truth', label = "truth")
    if (pl==nothing)
        display(pl1)
    end
    display(loss)
end

cb(d::datum) = cb(truth=d.labels,losspred=()->losspred(d),t=d.t)


function cb(b::batch)
    pl = Plots.plot()
    for i = 1 #in 1:size(b.u)[2] # only plot first five cases in batch b
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
    t = range(tspan..., length = 100)
    truth = neural_ode(dudt, u0, (0, stopTime))
    u1 = truth(startTime)
    n_ode(u) = neural_ode(dudt_train, [u[1:2]...,pred_bears(truth[1:2,1:6])...], tspan, Tsit5(), saveat = t)
    truth2 = neural_ode(dudt, u1, tspan)
    labels = Flux.data(truth2(t))
    datum(n_ode, labels, u1, t)
end

function uDatum(u0, dur)
    tspan = (0.0f0, dur)
    t = range(tspan..., length = 100)
    n_ode2(u) = n_ode(u, t=t, tspan=tspan)
    truth2 = neural_ode(dudt, u0, tspan)
    labels = Flux.data(truth2(t))
    datum(n_ode2, labels, u0, t)
end

function uBatch(u0s, dur)
    tspan = (0.0f0, dur)
    t = range(tspan..., length = 100)
    truth = neural_ode(dudt, u0s, tspan)
    labels = Flux.data(truth(t))
    n_ode2(u) = n_ode(u, t=t, tspan=tspan)
    batch(n_ode2, labels, u0s, t)
end


Random.seed!(2)
dataInits = [
        (uDatum(6.0f0*rand(Float32,size(u0)...), 3.0f0),)
    for i in range(0.0f0, 6.0f0, length=8)]

inits(a) = hcat([
        (rand(Float32,size(u0)).+Float32(a))
    for i in 1:8]...)
dataBatch = [
        (uBatch(inits(a),5.0f0),)
    for a in 1:20]


cb(dataBatch[1]...)

@Flux.epochs 50 Flux.train!(
    loss,
    tracking,
    dataBatch,
    ADAM(1.0E-3);
    cb = Flux.throttle(()->cb(dataBatch[1]...), 3),
)

#
