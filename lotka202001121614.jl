
# adding in a bypass network for first derivative

# plans
# add in parallel problems into loss for mini batching
# try solving just the prediction problme without IDE
# create fast testing set based on prediction of derivatives at each u0, not solving
# ode.
# retain best results on test set.



using Flux  #DiffEqFlux, DifferentialEquations
using DiffEqFlux
using DifferentialEquations # need Tsit5
using LinearAlgebra
import Plots
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


# end
println("Loaded Packages")

#import Flux
struct lv
    A
    b
end

function mk_lv(A::AbstractArray,b::AbstractArray)
    lv( Flux.param(A), Flux.param(b))
end

(m::lv)(x) = x.*( m.b .+ m.A*x)

@Flux.treelike lv

# The true Lotk Volterra model
α= 1.3
β= 0.9
γ=0.5
δ= 1.8
A1 = Float32[ 0.0 -β  -0.1; γ 0.0 -0.1 ; 0.3 0.98 0.0]
b1 = Float32[ α, -δ*0.5, -2.0] # netter with bears
#b1 = Float32[ α, -δ, -2.0]
# the true RHS of ODE
dudtTrue = lv(A1,b1)

# required interface template for ODEProblem
# could make this a method of lv instead.
function lotka_volterra(du,u,p,t)
    du .= dudtTrue(u)
end

# initial condition
u0 = Float32[ 0.44249; 4.6280594;0.5]
u0 = Float32[ 2.0; 7.0;0.0]  #1.0
#define timespan of interest
tspanFull = (0.0f0,80.0f0)
tFull = range(tspanFull...,length=500)

# set up and solve ODE problem
prob = ODEProblem(lotka_volterra,u0,tspanFull,())
sol = solve(prob)

#Now at this point sol is fully capable of interpolation

# this creates a pair of time vector (.t) and 3-element point vector .p
labelsTrue = sol(tFull)
Plots.scatter(labelsTrue,markersize= 1,label="Truth")
Plots.plot!(sol,label="Truth")
# pull out just the points without the time axis
lablesFull = labelsTrue.u

#####
# now were turn to the neural net
#####

###
# make the input vector
# this vector is the initial points augmented with initial slopes
# however we restrict this to just 2 animals and sheer off the third
M = size(u0)[1]
get_uA(u0) = vcat( u0[1:2], dudtTrue(u0)[1:2])
get_uA_at_t(t;method=sol) = get_uA(sol(t))
uA = get_uA(u0)
# note than in the future might also inclued t0 as input.

N=size(uA)[1]
###

#the neual ODE
function resetModel()
    global dudt_train1,dudt_train0 ,tracking
    W=40
    W2 = Int(W/2)
    g(N,w) = ForkCombine(
       Dense(N,Int(w/4),tanh),
       Dense(N,Int(3*w/4),swish),
        vcat ) # this creates these if they do not exist.

    dudt_train0 = Chain(
       g(N,W),g(W,W),g(W,W2), g(W2,W2),g(W2,W2),#,
      Dense(W2,Int(N/2))
            )
    dudt_train1 = ForkCombine( x->Float32[0 0 1 0; 0 0 0 1]*x, dudt_train0, vcat)
 #Flux.SkipConnection(dudt_train0,(m,x)->vcat(x,m))
    tracking = Flux.params(dudt_train1)
end

resetModel()
dudt_train1(uA)
dudt_train1
function revert!(dudt=dudt_train1,tracking=tracking)
      if loss_best_deriv != nothing
          dudt = deepcopy( loss_best_deriv)
          tracking = Flux.params(dudt)
      else
          @warn "Nothing to revert to"
      end
end
Nparams1 = N*32+32+32*32+32+32*32+32+32*N+N
###
#time spans to fit training data
###
pnts=30
function mk_labels(;method=sol,pnts=30,endTime=7.0f0,dim=N, startTime=0.0f0)
    tspan0 = (startTime, endTime)
    t0 = range(tspan0...,length=pnts)
#labels = makeTruth(u0,dudt1,tspan=tspan,t=t)  # untracked?
#abels = Flux.data(labels)
    labels0 = zeros(Float32,N,pnts)
    tmp = hcat(method(t0).u...)  # turns an array of arrays into  2D array note ...
    labels0[1:2,:] = tmp[1:2,:]


    tspan0,t0,labels0
end

tspan0,t0,labels0 = mk_labels(;pnts=pnts,endTime=7.0f0)
#show what we got
Plots.scatter!(t0,labels0',label="points")

# Define the neural ODE over this time range.
n_ode0 = x->neural_ode(dudt_train1,x,tspan0,Tsit5(),saveat=t0) #p,reltol=1e-7,abstol=1e-9)
 # Test if it works
n_ode0(uA) # this is only a test

function makePred(uA,method = dudt_train1; t=tFull,tspan=tspanFull)
    n_ode_truth = x->neural_ode(method,x,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9) #|> gpu
    n_ode_truth(uA) #|> gpu#.data # Get the prediction using the correct initial condition
 end
###
#  Loss Functions
###

# compute L1 of all the parameters. used for sparsity control
function sparseL1(;params=tracking, Nparams=Nparams1)
    sum(map(x->sum(abs.(x)), params))/Nparams  # does collect strip the Tracking?
end

struct LossLabels
    cmap::AbstractArray{Real} # no longer in use!
    #regularizer::AbstractArray{Real}
end


# note prod(size(labels)) needs to be untracked!!!!!!!!
(m::LossLabels)(pred::AbstractArray{Float32,3}, labels::AbstractArray{Float32,3} ) =
 (sum( abs2,(pred-labels),dims=(2,3)))./length(labels) #length must not be tracked.
   # note: in the future if we have variable length traing sets in a collection then
   # we may want to divide each collection by it's size.
   # could precalc log(lables0)
#(m::LossLabels)(pred::AbstractArray{Float32,3}, labels::AbstractArray{Float32,3} ) =
#    (sum( -pred.*log(labels),dims=(2,3)))./length(labels)
# note we want the u to be cross entropy and the u dot to be ignored or msr from 0
 #length must not be tracked.
      # note: in the future if we have variable length traing sets in a collection then
      # we m
(m::LossLabels)(pred::AbstractArray{Float32,2}, labels::AbstractArray{Float32,2}) =
  (sum( abs2,(pred-labels),dims=2))./length(labels);
#(m::LossLabels)(pred::AbstractArray{Float32,2}, labels::AbstractArray{Float32,2}) =
#    (sum(-pred.*log(labels),dims=2))./length(labels);

directLoss=  LossLabels(Float32[1.0 1.0 0.0 0.0; 0.0 0.0 1.0 1.0])


dataLoss(pred,  labels) = directLoss(pred, labels )

n_ode0 = x->neural_ode(dudt_train1,x,tspan0,Tsit5(),saveat=t0) #p,reltol=1e-7,abstol=1e-9)
 # Test if it works
n_ode0(uA)

struct datum0  # wraps up a nice package of n_ode for prediciton and Labels
    n_ode::Function  # the ode
    uA::AbstractArray{Float32} # the initial condition input for n_ode
    labels::AbstractArray{Float32} # the True answer
    t # the time range corresponding to the points
end



function loss( d::datum0; dloss=dataLoss,
      weightS=0.0f0, weightR = [1.0f0, 1.0f0, 0.0005f0, 0.0005f0], tracking=tracking)

    pred = d.n_ode(d.uA)
    s=sum( weightR.*dloss(pred,d.labels)) + weightS*sparseL1(params=tracking)
    return s
end

data0 = [(datum0(n_ode0,uA,labels0,t0),),] # Array of tuples
loss(data0[1]...)

import Dates
D=Dates.format(Dates.now(), "yyyymmddHHMMSS")

fileTag="/tmp/nobear_"*D
function get_dated_file(filename=fileTag)
    fileTag*Dates.format(Dates.now(), "yyyymmddHHMMSS")
end
get_dated_file()


function TrainIt(it,opt,data;loss=loss, cb=cbp, tracking=tracking)
    println("training ",it, " cycles")
    #data = Iterators.repeated(data,it)
#    opx = Flux.NADAM(opt) # AMSGrad(opt)

    try
    @Flux.epochs it Flux.train!(loss, tracking, Random.shuffle!(data), opt; cb=cb)
    catch # reset to last best result
        if loss_best_deriv != nothing
            revert!()
        end
        @warn "Caught failure... moving on"
    end#Flux.throttle(cbp,8))
end


function extrap(;filename=nothing,tspan2=tspanFull,t2=tFull,truth=sol)
    global fileTag
    labels2 = sol(t2) # makeTruth(u0,dudt1;t=t2,tspan=tspan2)
    pred_full = Flux.data(makePred(uA,dudt_train1;t=t2,tspan=tspan2))
    pl =Plots.plot(layout=(2,1))
    Plots.scatter!(pl,t2,pred_full[1:2:end,:]',layout=(2,1),label="pred",markersize=[ 3 3 2 2], color = ["orange"  "orange" "green" "red"])
    Plots.scatter!(pl,t2,pred_full[2:2:end,:]',layout=(2,1),label="pred",markersize=[ 3 3 2 2], color = ["lightblue"  "lightblue" "green" "grey"])

    Plots.plot!(pl,labels2,label="data",
        ylims = (0,20), linewidth=[ 3 3 2 2],
        color = ["orange" "lightblue" "green" "grey"];
        subplot=1)

    if filename != nothing
        Plots.pdf(pl,filename)
    else
        display(pl)
    end
end
extrap()


# useful commands
# methods() fieldnames() @code_lowered
nEpoch = 0
loss_best_deriv = nothing
loss_best = 5000.0f0

resetModel()

cbp = function(;force = false, dd=data0,
      dataLoss=dataLoss,
      loss=loss,
       tracking=tracking,
        filename=nothing,
        tcount=20)

    global nEpoch,fileTag
    print
    if (!force)
        nEpoch += 1
        if  nEpoch%tcount != 0
        return
        end
    end
    println("---- ",nEpoch)

    spl = Flux.data(sparseL1(params=tracking))

    pl = Plots.plot(layout=(2,2),legend=false)
    total_ls =0.0f0
    for ddd in dd
        d=ddd[1]

        pred=Flux.data(d.n_ode(d.uA))
        dl= dataLoss(pred, d.labels)
        ls = sum(dl[1:2])
        rs = sum(dl[3:4])
        la = loss(d)
        Plots.scatter!(pl,d.t,pred[1:2,:]',
        label="pred",markersize=[ 2 2 1 1 ], color = ["orange" "lightblue" "green" "grey"];
        )
        Plots.scatter!(pl,d.t,pred[3:3,:]',
        label="hidden",markersize=2, xlim=tspanFull,color = ["orange" "lightblue" "green" "grey"];
        subplot=3)
        Plots.scatter!(pl,d.t,pred[4:4,:]',
        label="hidden",markersize=2, xlim=tspanFull,color = ["lightblue" "green" "grey"];
        subplot=4)
        # Plots.plot!(pl,d.t,d.labels',ylim=(-3.0f0,20.0f0),label="truth",
        # linewidth=[ 2 2 1 1], color = ["orange" "lightblue" "green" "grey"];
        # subplot=1)
        # Plots.plot!(pl,d.t,d.labels',ylim=(-3.0f0,20.0f0),label="truth",
        # linewidth=[ 2 2 2 2], color = ["orange" "lightblue" "green" "grey"];
        # subplot=2)
        Plots.plot!(pl,sol,ylim=(-3.0f0,20.0f0),label="truth",
        linewidth=[ 1.5 1.5 1 1], color = ["orange" "lightblue" "green" "grey"];
        subplot=1)
        Plots.plot!(pl,sol,ylim=(-3.0f0,20.0f0),label="truth",
        linewidth=[ 1.5 1.5 1 1], color = ["orange" "lightblue" "green" "grey"];
        subplot=2)
        totalLoss = ls+rs+spl

        println("LOSS ",la," ls ",ls," rs ",rs," dl ", dl," sparse ",spl,  " TL ",totalLoss)
        total_ls += ls
    end
        println("summed losses $total_ls")
    if filename==nothing
        display(pl) # needed?
    else
        Plots.pdf(pl, filename)
    end


    if loss_best>total_ls && !force
        global loss_best, loss_best_deriv, dudt_train1
        loss_best= total_ls
        loss_best_deriv=deepcopy(dudt_train1)
        println("best so far")
    end
    total_ls
end

#cb1(Flux.throttle(cbp,3))
cbp(force=true,tcount=1)
loss1(d) = loss(d ;
      weightS=0.0f0, weightR = [1.0f0, 1.0f0, 0.0001f0, 0.0001f0], tracking=tracking)

loss1(data0[1]...)

loss3(d) = loss(d ;
      weightS=0.0f0, weightR = [1.0f0, 1.0f0, 0.00000f0, 0.00000f0], tracking=tracking)

loss3(data0[1]...)


function make_datum(startTime::Float32,endTime::Float32,pnts::Int)
    tspan1,t1,labels1 = mk_labels(;method=sol,pnts=pnts,endTime=endTime,dim=N, startTime=startTime)
#labels1 = makeTruth(u0,dudtTrue,tspan=tspan1,t=t1)  # untracked?
    labels1 = Flux.data(labels1)
    Plots.scatter!(t1,labels1',label="points")
    n_ode1 = x->neural_ode(dudt_train1,x,tspan1,Tsit5(),saveat=t1)
    (datum0(n_ode1,get_uA_at_t(startTime),labels1,t1),)  # array of tuples # possibly should make uA a copy?
end

data2a = [ make_datum(i+0.0f0,i+4.5f0,30) for i in range(0.0f0,23,length=15)]
data2b = [ make_datum(i+0.0f0,i+4.5f0,30) for i in range(40.0f0,53f0,length=10)]
data2 = vcat(data2a,data2b)
import Random
Random.shuffle!(data2)
cbp2(;force=false, filename=nothing) =
    cbp(;force = force,
    dd=data2,
      dataLoss=dataLoss,
      loss=loss1,
      # tracking=tracking,
        filename=filename,
        tcount=90
    )

cbp2(force=true)
extrap()
cbp2(force=true,filename=get_dated_file())
extrap(filename = get_dated_file())
cbp2(force=true)
optx = ADAM(1E-2)
loss_best=5000


function massage(epochs,maxcounts,loss_goal,optx; loss1=loss1,cb2=cbp2)
    icount=0
    while loss_best>0.1 && icount<maxcounts
        TrainIt(epochs,optx,data2; tracking=tracking, loss=loss1, cb=cbp2)
        revert!()
        extrap()
        cbp2(force=true,filename=get_dated_file())
        extrap(filename = get_dated_file())
        icount+=1
    end
    println("==========================switcing $icount best $loss_best")

end
nEpoch=1
loss_best=5000
loss_best_deriv=nothing
optx=ADAM(1E-3)
massage(30,5,10,optx)
massage(30,5,10,optx)
optx=ADAM(1E-6)
massage(30,5,10,optx)
optd=Descent(1E-8)
massage(50,8,0.001,optd)
optd=Descent(1E-9)
massage(50,8,0.0001,optd)

cbp3(;force=false, filename=nothing) =
    cbp(;force = force,
    dd=data2,
      dataLoss=dataLoss,
      loss=loss3,
      # tracking=tracking,
        filename=filename,
        tcount=90
    )
