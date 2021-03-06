using DifferentialEquations, Flux, Plots, DiffEqFlux, LinearAlgebra
# function lotka_volterra(du,u,p,t)
#   x,y = u
#   a, b, c, d = p
#   du[1] = dx =  a*x - b*x*y
#   du[2] = dy = -c*y + d*x*y
# end
# u0 = [0.44249; 4.6280594]#[1.0f0,1.0f0]


tspan = (0.0f0,16.0f0)
t = range(tspan[1], tspan[2], length=100)
# p = [1.3f0  0.9f0
#      1.8f0  0.5f0]
# prob = ODEProblem(lotka_volterra,u0,tspan,p)
include("MyUtils.jl")





# truth ODE
## 2-animal lotka-volterra
u0 = gc(Float32[0.44249; 4.6280594])
A = Float32[0.0 -0.9; 0.5 0.0]
b = Float32[1.3, -1.8]
dudt = lv(A, b)

sol = neural_ode(dudt, u0, tspan, saveat=t)#solve(prob,Tsit5(),saveat=t)
data = Array(Flux.data(sol))

plot(t, data')


struct LV
    ac
    bd
    flip
        # LV(ac, bd, flip) = new(Flux.param(ac), Flux.param(bd), flip)
end

LV(ac, bd) = LV(Flux.param(ac), Flux.param(bd),
                Matrix{Float32}(I, size(ac)[1], size(ac)[1])[:,end:-1:1])

(m::LV)(x) = (m.ac .+ m.bd.*(m.flip*x)).*x

Flux.@treelike LV
#
# p = [1.2f0 -1.2f0; -2.7f0 0.8f0] # initial geuss

# dudt = Chain(LV([1.2f0; -3.3f0],[-0.7f0; 1.1f0]))
# Chain(
#             LV,
#             # to start, the network is the identity matrix, so the initial geuss it just the simple ODE alone.
#             LinNonlin(Matrix{Float32}(I, 4, 2), convert(Array{Float32,1}, zeros(4)), [0.0f0], tanh),
#             LinNonlin(Matrix{Float32}(I, 6, 4), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
#             # LinNonlin(Matrix{Float32}(I, 6, 6), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
#             # LinNonlin(Matrix{Float32}(I, 6, 6), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
#             # LinNonlin(Matrix{Float32}(I, 6, 6), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
#             # LinNonlin(Matrix{Float32}(I, 6, 6), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
#             # LinNonlin(Matrix{Float32}(I, 6, 6), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
#             LinNonlin(Matrix{Float32}(I, 6, 6), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
#             LinNonlin(Matrix{Float32}(I, 3, 6), convert(Array{Float32,1}, zeros(3)), [0.0f0], tanh),
#             LinNonlin(Matrix{Float32}(I, 2, 3), convert(Array{Float32,1}, zeros(2)), [0.0f0], tanh)
# )
dudt_train = Chain(
    Dense(2, 32, swish),
    Dense(32, 32, swish),
    Dense(32, 32, swish),
    Dense(32, 2),
)
ps = Flux.params(dudt_train)

n_ode = x->neural_ode(dudt_train,x,tspan,Tsit5(),saveat=t)#,reltol=1e-7,abstol=1e-9)

pred = n_ode(u0)

predict_n_ode(u=u0) = n_ode(u)

loss_n_ode() = sum(abs2, data .- predict_n_ode())

cb = function () # callback function to observe training
    display(loss_n_ode())
    # display(("ps",ps))
    # plot current prediction against data
    cur_pred = Flux.data(predict_n_ode())
    pl = Plots.plot()
    t2 = range(tspan[1], tspan[2], length=10)
    plot!(pl,t,data',label="data")
    scatter!(pl,t,cur_pred',label="prediction")
    # plot!(t,cur_pred[1,:]-pendulumData[1,:],label="difference",linestyle=:dot)

    display(plot(pl))
end

cb()

function train(iters::Int = 3; opt = ADAM(0.1), callback = cb, ps = ps, loss_n_ode = loss_n_ode)
    callback()
    d = Iterators.repeated((), iters)
    Flux.train!(loss_n_ode, ps, d, opt, cb = Flux.throttle(callback, 3))
end

train(30,opt=ADAM(0.0001))
