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

println(gc)



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



# set up of first case (2d lotka volterra)
u0 = gc(Float32[ 0.44249; 4.6280594])
α, β, γ, δ = 1.3, 0.9, 0.5, 1.8
A1 = Float32[ 0.0 -β ; γ 0.0]
b1 = Float32[ α, -δ]
dudt1 = lv(A1,b1)

# empty neural net of swishes
dudt_train1 = Chain(Dense(2,32,swish),Dense(32,32,swish),Dense(32,32,swish),Dense(32,2))
tracking = Flux.params(dudt_train1)

# could run on gpu
if cu == gc
    dudt1 = dudt1 |> gpu
    dudt_train1 = dudt_train1 |> gpu
end

# neural net solver
n_ode(u) = neural_ode(dudt_train1,u,tspan,Tsit5(),saveat=t) #p,reltol=1e-7,abstol=1e-9)

# loss function
loss(;n_ode=n_ode,truth=truth)=sum(abs2, (n_ode(u0)-truth))# .*[1.0f0 1.0f0 0.0f0]' )

# do loss but dont throw away prediction
function faster_loss(;n_ode=n_ode,truth=truth)
    a = Flux.data(n_ode(u0))
    loss = sum(abs2, (a-truth) )#.*[1.0f0 1.0f0 0.0f0]' )
    return loss, a
end

# plot prediction and truth
cb = function ()
    ls,cur_pred = faster_loss()
    pl = Plots.plot(t,truth',label="truth")
    Plots.scatter!(pl,t,cur_pred',label="prediction")
    display(pl)
    println(ls)
end

# train nn to match solution in given interval
tspan = (0.0f0,3.0f0)
t = range(tspan...,length=100)

# recalculate truth and plot it against prediction
truth = neural_ode(dudt1,u0,tspan,saveat=t)
Plots.plot(t,truth')
cb()

# train first with adam, then descent
Flux.train!(loss, tracking, Iterators.repeated((),500), ADAM(2.0E-3); cb = Flux.throttle(cb, 3))
Flux.train!(loss, tracking, Iterators.repeated((),500), Descent(1.0E-6); cb = Flux.throttle(cb, 3))

# again for larger interval
tspan = (0.0f0,18.0f0)








# 3-animal version

# parameters and real dudt
u0 = gc(Float32[ 2.5; 5.0; 7.5 ])
aa,bb,cc,dd,ee,ff,gg,hh,ii = 0.4, 0.535, 0.532, 0.531, 0.2, 0.536, 0.534, 0.533, 0.2
A1 = Float32[ 0.0  -bb   cc ;
               dd  0.0  -ff ;
              -gg  hh  0.0 ]
b1 = Float32[ aa, -ee, -ii ]
dudt1 = lv(A1,b1)

# empty neural net of swishes
dudt_train3 = Chain(Dense(3,32,swish),Dense(32,32,swish),Dense(32,32,swish),Dense(32,3))
tracking = Flux.params(dudt_train3)
# neural net solver
n_ode(u) = neural_ode(dudt_train3,u,tspan,Tsit5(),saveat=t) #p,reltol=1e-7,abstol=1e-9)

# set small timespan and train
tspan = (0.0f0,1.0f0)
t = range(tspan...,length=100)

# recalculate truth and plot it against prediction
truth = neural_ode(dudt1,u0,tspan,saveat=t)
Plots.plot(t,truth')
cb()

# train first with adam, then descent
Flux.train!(loss, tracking, Iterators.repeated((),500), ADAM(1.0E-5); cb = Flux.throttle(cb, 3))
Flux.train!(loss, tracking, Iterators.repeated((),100), Descent(1.0E-7); cb = Flux.throttle(cb, 3))

tspan = (0.0f0,10.0f0)



# 3-animal lotka-volterra, one is hidden from the nn

# loss function - only on first two channels
loss(;n_ode=n_ode,truth=truth)=sum(abs2, (n_ode(u0)[1:2,:]-truth[1:2,:]))# .*[1.0f0 1.0f0 0.0f0]' )

# do loss but dont throw away prediction
function faster_loss(;n_ode=n_ode,truth=truth)
    a = Flux.data(n_ode(u0))
    loss = sum(abs2, (a[1:2,:]-truth[1:2,:]) )#.*[1.0f0 1.0f0 0.0f0]' )
    return loss, a
end
