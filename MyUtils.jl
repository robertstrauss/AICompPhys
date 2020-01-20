using Flux, DiffEqFlux, DifferentialEquations, LinearAlgebra

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
