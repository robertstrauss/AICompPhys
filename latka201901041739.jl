
using CuArrays
using Flux  #DiffEqFlux, DifferentialEquations
using DiffEqFlux
using DifferentialEquations # need Tsit5
gc = cu
cu = 0
gc = Array

if cu==gc
    CuArrays.allowscalar(false)

    CuArrays.allowscalar(true)
end

println(gc)
#module mbl
function eye(n,m)
     foo = zeros( Float32,m,n ) #|> gpu
     for i in 1:min(n,m)
       foo[i,i]=1.0f0
     end
     foo
end
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
# m = Chain(Dense(2,3), IdentitySkip(Dense(3, 3)), Dense(3,4))
#Chain(Dense(2, 3), IdentitySkip(Dense(3, 3)), Dense(3, 4))


struct coshLayer
  c::TrackedArray
end


function coshLayer(W::AbstractArray;gc=gc)
    x = Flux.param(gc(W))
    coshLayer(x)
end

function (a::coshLayer)(x::AbstractArray)
   x.*x.*x.*x.*x
end

#Flux.@treelike sinhLayer
struct sinhLayer
  c::TrackedArray
end

function sinhLayer(W::AbstractArray;gc=gc)
    x=Flux.param(gc(W)) #|>gpu
    sinhLayer(x)
end

function (a::sinhLayer)(x::AbstractArray)
   x.*x.*x
end
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
function myDense2(activation::Function,W::AbstractArray ,b::AbstractArray; gc=gc)
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
#define the true data set

# end module


# for any type that has @treelike
# mapleaves(cu, thing)
# will convert every tracked parameter to GPU
# however it will not convert untracked parameters!

#module dh
function deeptanh(;gc=gc)
    #simplePendulum = gc(Float32[0.0 1.0 ; -9.8 0.0])
     select2 = gc(Float32[0.0 1.0 ] )
     select1 = gc(Float32[1.0 0.0 ] )
    #ann =  JoinLayers( x->select2*x,Chain(x->select1*x,Dense(1,1,sin),Dense(1,1) ))
    ann =  JoinLayers( x->select2*x,Chain(x->select1*x,myDense2(sin,[  -0.8258633f0]',[0.0f0]),myDense2([5.320666f0]',[0.0f0]) ))
    bann =JoinLayers( x->select2*x,Chain(
        Dense(2,16,tanh),
        JoinLayers(Dense(16,8,tanh),
                   JoinLayers(  Dense(16,4,cos), Dense(16,4,sin))
                   ),

        Dense(16,8,tanh),
        Dense(8,8,tanh),
        Dense(8,1)
        )

        )
#    g = x->[x[2] ; ann(x)[1]]
    #println(dudt)
    ps = Flux.params(ann)
    return ann,ps
end

# end module




println("foo")

#import Flux
struct lv
    A
    b
    dummy
end

function mk_lv(A::AbstractArray,b::AbstractArray)
    lv( Flux.param(A), Flux.param(b),1.0)
end

(m::lv)(x) = x.*( m.b .+ m.A*x)

@Flux.treelike lv



#import .lv


#u0 = gc(Float32[pi/4,0.0])

tspan = (0.0f0,30.0f0)
t = range(tspan...,length=100)

u0 = gc(Float32[ 3.14*0.95; 2.0])
u0 = gc(Float32[ 3.14*0.95; 2.0; 1.0])
  ###
  # create the labels
  ###
#select2 = gc(Float32[0.0  1.0])
#select1 = gc(Float32[1.0  0.0])
#dudt1 = mbl.JoinLayers( x->select2*x, x->-9.8*(sin.(select1*x)))
A1 = Float32[ 0.0 -2.0 1.0; 2.0 0.0 -1.0 ; 1.0 -1.0 0.0]
b1 = Float32[ 1.0, -1.0, -0.1]
dudt1 = lv(A1,b1,1.0)
if cu == gc
    dudt1 =dudt1 |> gpu
end

function makeTruth(u0,method = dudt1; t=t,tspan=tspan)
    n_ode_truth = x->neural_ode(method,x,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9) #|> gpu
    n_ode_truth(u0) #|> gpu#.data # Get the prediction using the correct initial condition
end

using LinearAlgebra
import Plots
labels = makeTruth(u0,dudt1)  # untracked?
labels = Flux.data(labels)
Plots.plot(t,labels',label="Truth")
Plots.scatter!(t,labels',label="points")

A2 =1.0f0*A1
b2 = 1.0f0*b1
dudt_train1 = mk_lv(A2,b2)
dudt_train1 = Chain(CatSkip(x->x.*Float32[0.0 1.0 ; 1.0 0.0]*x),
                   myDense2(Float32[ b2[1] 0.0 0.0 A2[1,2] ; 0.0 b2[2] A2[2,1] 0.0]),
                   x->5.0f0.*Flux.tanh.(x./5.0f0)
                   )
dudt_train1 = Chain(ForkCombine( myDense2(Float32[ b2[1] 0.0 0.0  ; 0.0 b2[2] 0.0  ; 0.0 0.0 0.0]),
                            x-> x.*myDense2(Float32[ 0.0 A2[1,2] 0.0 ; A2[2,1] 0.0 0.0;  0.0 0.0 0.0 ;])(x),
                            (x,y) -> x.+y),
                x->5.0f0.*Flux.tanh.(x./5.0f0)
                                      )
u0 = gc(Float32[ 3.14*0.95; 2.0; 1.0])
if gc == cu
    dudt_train1 = dudt_train1 |> gpu

end
#tracking = params(dudt_train1)
tracking = Flux.params(dudt_train1)

n_ode = x->neural_ode(dudt_train1,x,tspan,Tsit5(),saveat=t) #p,reltol=1e-7,abstol=1e-9)

  # show that the problem isn't in neural_ode by itself
pred = n_ode(u0)

function loss()
      sum(abs2, (labels.+3.0f0).*(n_ode(u0)-labels).*[1.0f0 1.0f0 0.0f0]' )
  end

function faster_loss()
    a = Flux.data(n_ode(u0))
    loss = sum(abs2, (labels.+3.0f0).*(a-labels).*[1.0f0 1.0f0 0.0f0]' )
    return loss,a
end

import Plots




loss_best = 1000.0f0
loss_best_params = []
foo = 0
cbp = function(;force = true)
    global foo
    foo += 1
    if (force)
        if  foo%10 != 0
        return
    end
    end
    println("---- ",foo)

#    println("ps ",tracking)

  # plot current prediction against data
    ls,cur_pred = faster_loss() # Flux.data(n_ode(u0))
    #pl =plot(t,ode_data[1,:],label="data")
    pl =Plots.plot(t,labels',label="data")
    #scatter!(pl,t,cur_pred[1,:],label="prediction")
    Plots.scatter!(pl,t,cur_pred',label="prediction")
    display(Plots.plot(pl))
    println("loss ",ls)
    if ls<loss_best
        global loss_best, lost_best_params, d
        loss_best= ls
        loss_best_deriv=deepcopy(dudt_train1)
        println("best so far")
    end
end




#cb1(Flux.throttle(cbp,3))
cbp(force=false)



function TrainIt(it,opt=0.01f0)
    println("training ",it, " cycles opt=",opt)
    data = Iterators.repeated((),it)
    opx = Flux.NADAM(opt) # AMSGrad(opt)
    Flux.train!(loss, tracking, data, opx; cb = cbp ) #Flux.throttle(cbp,8))
end

loss_best = 5000.0f0
loss_best_params = []

TrainIt(20,0.1)
TrainIt(100,0.01)
TrainIt(30,0.003)
TrainIt(100,0.003)
TrainIt(20,0.0003)
TrainIt(100,0.0003)
TrainIt(300,0.0003)
TrainIt(20,0.00003)
TrainIt(100,0.00003)
#TrainIt(4,0.0001)
TrainIt(512,0.00003)
TrainIt(1280,0.00001)
println(dudt.W)
println(dudt_train.W)


Chain(
    JoinLayers( x->select2.*x,
    Chain(
      CatSkip(
        JoinLayers(coshLayer(eye(2,2)),sinhLayer(eye(2,2)))

        )      ,  #x->2.0f0.*Flux.tanh.(x./2.0f0),
      myDense2(identity,Float32[ 1.0 0.0 0.0 0.0 0.01 0.0;
              0.0 1.0 0.0 0.0 0.0 0.0], [1.0f0] ),

      #x->hh*x,
      CatSkip(
      JoinLayers(coshLayer(eye(2,2)),sinhLayer(eye(2,2)))
      )
        ,
      myDense2(identity,Float32[ 1.0 0.0 0.0 0.0 0.0 0.0;
            0.0 1.0 0.0 0.0 0.0 0.0], [1.0f0] )
            ,
 #x->select2*x,
          )
 ),
  # the following is just a parallel efficienct minmax
  x->15.0f0.*Flux.tanh.(x./15.0f0),

#  minmaxLayer(-10,10)
  )
