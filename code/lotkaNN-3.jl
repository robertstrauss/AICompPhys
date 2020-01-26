using Flux, DiffEqFlux, DifferentialEquations, LinearAlgebra
import Plots


# GPU
cu = 0
gc = Array

if cu==gc
    using CuArrays
    CuArrays.allowscalar(false)

    CuArrays.allowscalar(true)
    gc = cu
end



# identity matrix
#module mbl
function eye(n,m)
     eye = zeros( Float32,m,n ) #|> gpu
     for i in 1:min(n,m)
       eye[i,i]=1.0f0
     end
     eye
end



# useful layers
struct IdentitySkip
    inner
end

struct CatSkip
    inner
end

struct JoinLayers
    inner1
    inner2
end

struct ForkCombine
    inner1
    inner2
    method::Function
end


(m::IdentitySkip)(x) = m.inner(x) .+ x

(m::CatSkip)(x) = vcat(x,m.inner(x))

(m::JoinLayers)(x) = vcat(m.inner1(x),m.inner2(x))

(m::ForkCombine)(x) = m.method(m.inner1(x),m.inner2(x))

Flux.@treelike CatSkip
Flux.@treelike IdentitySkip
Flux.@treelike JoinLayers
Flux.@treelike ForkCombine



struct coshLayer
  c::TrackedArray
end

coshLayer(W::AbstractArray;gc=gc) = coshLayer(Flux.param(gc(W)))

(a::coshLayer)(x::AbstractArray) = x.*x.*x.*x.*x

struct sinhLayer
  c::TrackedArray
end

sinhLayer(W::AbstractArray;gc=gc) = sinhLayer((Flux.param(gc(W))))

(a::sinhLayer)(x::AbstractArray) = x.*x.*x

Flux.@treelike sinhLayer
Flux.@treelike coshLayer



struct myDense2
  W::AbstractArray
  b::AbstractArray
  beta::AbstractArray # need union of Array and Tracked Arrya
  activation::Function
end

function myDense2(activation::Function,W::AbstractArray, b::AbstractArray, beta::Real ; gc = gc)
    p = map( gc,(W,b,beta) )
    p = map(Flux.param, p)
    myDense2(p...,activation)# cast the array to a tracked parameter list
                            # then create and return the struct
end

function myDense2(activation::Function, W::AbstractArray, beta::Real ; gc=gc)
    b = zeros(Float32,size(W)[1])
    W,b,beta = map( gc, (W,b,beta))
    W,beta = map(Flux.param, (W,beta))  # b is not tracked
   # but b is on gpu possibly
    myDense2(W,b,beta,activation )# cast the array to a tracked parameter list
                            # then create and return the struct
end

function myDense2(activation::Function,W::AbstractArray, b::AbstractArray; gc=gc)
    beta = [1.0f0]
    W,b,beta = map( gc,(W,b,beta) )
    W,beta = map(Flux.param, (W,beta))  # note comma on unpack
    myDense2(W,b,beta,activation)# cast the array to a tracked parameter list
                                 # then create and return the struct
end

function myDense2(W::AbstractArray ,b::AbstractArray; gc=gc)
    beta = [1.0f0]
    W,b,beta = map( gc,(W,b,beta) )
    W,beta = map(Flux.param, (W,beta))  # note comma on unpack
    myDense2(W,b,beta,identity)# cast the array to a tracked parameter list
                               # then create and return the struct
end

function myDense2(W::AbstractArray; gc=gc)
    beta = [1.0f0]
    b = zeros(Float32,size(W)[1])
    W,b,beta = map( gc,(W,b,beta) )
    W, = map(Flux.param, (W,))  # note comma on unpack
    myDense2(W,b,beta,identity)# cast the array to a tracked parameter list
                               # then create and return the struct
end

function (a::myDense2)(x)
    u = a.W*x .+ a.b
    a.beta.*a.activation.(u)
end


Flux.@treelike myDense2



## Lotka-Volterra layer
struct lv
    A
    b
end

(m::lv)(x) = x.*( m.b .+ m.A*x)

@Flux.treelike lv

function mk_lv(A::AbstractArray,b::AbstractArray)
    lv(Flux.param(A), Flux.param(b))
end










# now onto the actual doing!





cmap1 = Float32[1.0]
cmap2 = Float32[1.0]

# truth ODE
## 2-animal lotka-volterra
u0 = gc(Float32[ 0.44249; 4.6280594])
A = Float32[ 0.0 -0.9 ; 0.5 0.0 ]
b = Float32[ 1.3, -1.8]
dudt = lv(A,b)

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







# empty neural net of swishes
## 2-animal
dudt_train = Chain(Dense(2,32,swish),Dense(32,32,swish),Dense(32,32,swish),Dense(32,2))
## 3-animal
dudt_train = Chain(Dense(3,32,swish),Dense(32,32,swish),Dense(32,32,swish),Dense(32,3))
## tracking, necessary to train
tracking = Flux.params(dudt_train)





# could run on gpu
if cu == gc
    dudt = dudt |> gpu
    dudt_train = dudt_train |> gpu
end




# neural net solver
n_ode(u) = neural_ode(dudt_train,u,tspan,Tsit5(),saveat=t) #,reltol=1e-7,abstol=1e-9)

# loss function
loss(;n_ode=n_ode,truth=truth)=sum(abs2, (n_ode(u0).*cmap1-truth.*cmap2))

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
@Flux.epochs 50 Flux.train!(loss2, tracking, data4, ADAM(1.0E-3) ;    cb = Flux.throttle(cb2, 3))
Flux.train!(loss, tracking, Iterators.repeated((),500), Descent(1.0E-6); cb = Flux.throttle(cb, 3))
