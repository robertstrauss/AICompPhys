using Flux, DiffEqFlux, DifferentialEquations, LinearAlgebra, Tracker
import Plots

include("MyUtils.jl")





# truth ODE
## 2-animal lotka-volterra
u0 = gc(Float32[ 0.44249; 4.6280594])
A = Float32[ 0.0 -0.9 ; 0.5 0.0 ]
b = Float32[ 1.3, -1.8]
dudt = lv(A,b)

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

## use all channels
cmap1 = Float32[1.0]
cmap2 = Float32[1.0]


# time span
tspan = (0.0f0,3.0f0)
t = range(tspan...,length=100)

# recalculate truth and plot it against prediction
truth = neural_ode(dudt,u0,tspan,saveat=t)
Plots.plot(t,truth')


# empty neural net of swishes
## 2-animal net
dudt_train = Chain(Dense(2,32,swish),Dense(32,32,swish),Dense(32,32,swish),Dense(32,2))
## 3-animal net
"dudt_train = Chain(Dense(3,32,swish),Dense(32,32,swish),Dense(32,32,swish),Dense(32,3))"
## tracking, necessary to train
tracking = Flux.params(dudt_train)


# could run on gpu
"""
if cu == gc
    dudt = dudt |> gpu
    dudt_train = dudt_train |> gpu
end
"""

# neural net solver
n_ode(u) = neural_ode(dudt_train,u,tspan,Tsit5(),saveat=t) #,reltol=1e-7,abstol=1e-9)


# for parallel cases in a batch
struct batch
    n_ode::Function
    labels::AbstractArray
    u::AbstractArray
    t
end

# single case "batch"
struct datum
    n_ode::Function #NeuralODE
    labels::AbstractArray
    u0::AbstractArray
    t
end


## loss functions

loss(;n_ode=n_ode,truth=truth)=sum(abs2, (n_ode(u0).*cmap1-truth.*cmap2))

loss2(d::datum) = sum(abs2, (d.n_ode(d.u0).*cmap1-d.labels.*cmap2))

loss3(x) = loss2(x...)

lossBatch(batch...)=sum( abs2, case.n_ode(case.u0).*cmap1 - case.labels.*cmap2 )

# do loss but dont throw away prediction
function losspred(;n_ode=n_ode,truth=truth)
    a = Flux.data(n_ode(u0))
    loss = sum(abs2, (a.*cmap1-truth.*cmap2))#.*[1.0f0 1.0f0 0.0f0]' )
    return loss, a
end



## callbacks for output to plot prediction and truth
cb = function ()
    loss, cur_pred = losspred()
    pl = Plots.plot(t,truth',label="truth")
    Plots.scatter!(pl,t,cur_pred',label="prediction")
    display(pl)
    display(loss)
end
cb()

cb2 = function(x=data5)
    pl = Plots.plot(legend=false)
    for (i,) in x
        cur_pred = Flux.data(i.n_ode(i.u0))
        loss = loss2(i)
        Plots.plot!(pl,i.labels'[:,1],i.labels'[:,2],label="truth")
        Plots.scatter!(pl,cur_pred'[:,1],cur_pred'[:,2],label="prediction")
    end
    display(pl)
    display(loss)
end
cb2()

cbBatch = function(batches=dataPara)
    x = batches[1]
    pl = Plots.plot(legend=false)
    for i in x
        cur_pred = Flux.data(i.n_ode(i.u0))
        loss = loss2(i)
        Plots.plot!(pl,i.labels'[:,1],i.labels'[:,2],label="truth")
        Plots.scatter!(pl,cur_pred'[:,1],cur_pred'[:,2],label="prediction")
    end
    display(pl)
    display(loss)
end
cbBatch()



d1 = datum(n_ode, truth, u0, t)
data1 = [(d1,)]
data2 = vcat(data1,data1)

function timeDatum(startTime::Float32,stopTime::Float32;u0=u0)
    tspan = (startTime,stopTime)
    t = range(tspan...,length=30)
    truth=neural_ode(dudt,u0,(0,stopTime))
    u1 = truth(startTime)
    n_ode2(u) = neural_ode(dudt_train,u,tspan,Tsit5(),saveat=t)
    truth2 = neural_ode(dudt,u1,tspan)
    labels = Flux.data(truth2(t))
    datum(n_ode2,labels,u1,t)
end

function uDatum(u0,dur)
    tspan = (0.0f0,dur)
    t = range(tspan...,length=30)
    u1 = u0
    n_ode2(u) = neural_ode(dudt_train,u,tspan,Tsit5(),saveat=t)
    truth2 = neural_ode(dudt,u1,tspan)
    labels = Flux.data(truth2(t))
    datum(n_ode2,labels,u1,t)
end

data4 = [ ( timeDatum( i*1.0f0, i*1.0f0+3.0f0 ) ,) for i in range(0.0f0, 6.0f0,length=8 ) ]

data5 = [ ( uDatum( [i*1.0f0, 3.5f0+3.0f0*sin(i*1.0f0)], 3.0f0 ) ,) for i in range(0.0f0, 6.0f0,length=8 ) ]

dataPara = [ ( uDatum( [ [i+a, 3.5f0+3.0f0*sin(i)+a] for i in range(0.0f0, 6.0f0,length=8 ) ] , 5.0f0 )  ,) for a in range(0.0f0, 1.0f0, length=10) ]


loss2(data5[1]...)

lossBatch(dataPara[1])

# train first with adam, then descent
@Flux.epochs 5 Flux.train!(loss2, tracking, data5, ADAM(1.0E-3) ;    cb = Flux.throttle(cbBatch, 3))
Flux.train!(loss, tracking, Iterators.repeated((),500), Descent(1.0E-6); cb = Flux.throttle(cb, 3))
