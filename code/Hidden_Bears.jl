using Flux, DiffEqFlux, DifferentialEquations, LinearAlgebra
import Plots



## Lotka-Volterra layer
struct lv
    A
    b
end

(m::lv)(x) = x.*( m.b .+ m.A*x)

@Flux.treelike lv



# truth ODE

## 3-animal lotka-volterra system
u0 = Array(Float32[ 2.5; 5.0; 7.5 ])
A = Float32[ 0.0   -0.535  0.532 ;
             0.531  0.0   -0.536 ;
            -0.534  0.533  0.0   ]
b = Float32[ 0.4, -0.2, -0.2 ]
dudt = lv(A,b)

## only train on two of three channels
cmap1 = [1.0,1.0,0.0]
cmap2 = [1.0,1.0,0.0]







# empty neural net of swishes
dudt_train = Chain(Dense(3,32,swish),Dense(32,32,swish),Dense(32,32,swish),Dense(32,3))
## attach autograd trackers
tracking = Flux.params(dudt_train)



# neural net solver
n_ode(u) = neural_ode(dudt_train,u,tspan,Tsit5(),saveat=t) #,reltol=1e-7,abstol=1e-9)

# loss function
loss(;n_ode=n_ode,truth=truth)= function(;n_ode=n_ode,truth=truth)
    pred = n_ode(u0)
    return sum(abs2, ( pred.*cmap1 - truth.*cmap2 ))
end

struct datum
    n_ode::Function #NeuralODE
    labels::AbstractArray
    u0::AbstractArray
    t
end

function loss2(d::datum)
    sum(abs2, (d.n_ode(d.u0).*cmap1-d.labels.*cmap2))
end

function loss3(x)
    loss2(x...)
end

# do loss but dont throw away prediction
function losspred(;n_ode=n_ode,truth=truth)
    a = Flux.data(n_ode(u0))
    loss = sum(abs2, (a.*cmap1-truth.*cmap2))#.*[1.0f0 1.0f0 0.0f0]' )
    return loss, a
end

# plot prediction and truth
cb = function ()
    loss, cur_pred = losspred()
    pl = Plots.plot(t,truth',label="truth")
    Plots.scatter!(pl,t,cur_pred',label="prediction")
    display(pl)
    display(loss)
end

cb2 = function(x=data4)
    pl = Plots.plot(legend=false)
    for (i,) in x
        cur_pred = Flux.data(i.n_ode(i.u0))
        loss = loss2(i)
        Plots.plot!(pl,i.t,i.labels',label="truth")
        Plots.scatter!(pl,i.t,cur_pred',label="prediction")
    end
        display(pl)
        display(loss)
end
cb2()
# time span
tspan = (0.0f0,3.0f0)
t = range(tspan...,length=100)




# recalculate truth and plot it against prediction
truth = neural_ode(dudt,u0,tspan,saveat=t)
Plots.plot(t,truth')
cb()

d1 = datum(n_ode, truth, u0, t)
data1 = [(d1,)]
data2 = vcat(data1,data1)
function makeDatum(startTime::Float32,stopTime::Float32;u0=u0)
    tspan = (startTime,stopTime)
    t = range(tspan...,length=30)
    truth=neural_ode(dudt,u0,(0,stopTime))
    u1 = truth(startTime)
    n_ode2(u) = neural_ode(dudt_train,u,tspan,Tsit5(),saveat=t)
    truth2 = neural_ode(dudt,u1,tspan)
    labels = Flux.data(truth2(t))
    datum(n_ode2,labels,u1,t)
end
data4 =[ (makeDatum(i*1.0f0,i*1.0f0+3.0f0),) for i in range(0.0f0,6.0f0,length=8)]

loss2(data4[1]...)

# train first with adam, then descent
@Flux.epochs 50 Flux.train!(
    loss2,
    tracking,
    data4,
    ADAM(1.0E-3);
    cb = Flux.throttle(cb2, 3)
)
Flux.train!(loss, tracking, Iterators.repeated((),500), Descent(1.0E-6); cb = Flux.throttle(cb, 3))
