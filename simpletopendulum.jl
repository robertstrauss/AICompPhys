using Flux, DiffEqFlux, DifferentialEquations, Plots, LinearAlgebra

# initial condition and time span

u0 = [0.985*pi, 0.0]

tspan = (0.0, 10.0)

t = range(tspan[1],tspan[2],length=100)

# simple harmonic oscilator ODE

function simpleODE(du, u, p, t; g=-9.8f0)
    du[1] = u[2]
    du[2] = g*u[1]
end

simpleProb = ODEProblem(simpleODE, u0, tspan)

plot(solve(simpleProb))

# pendulum ODE (data to match)

function pendulumODE(du, u, p, t; g=-9.8f0)
    du[1] = u[2]
    du[2] = g*sin(u[1])
end

pendulumProb = ODEProblem(pendulumODE, u0, tspan)

pendulumData = Array(solve(pendulumProb, saveat=t))

plot(pendulumData')

# affine transformation plus nonlinear component

struct LinNonlin
    W
    b
    beta
    activation::Function
    LinNonlin(W, b, beta, activation) = new(Flux.param(W), Flux.param(b), Flux.param(beta), activation)
end

(ln::LinNonlin)(x::AbstractArray) = (u=ln.W*x+ln.b; ln.beta.*ln.activation.(u)+u)

Flux.@treelike LinNonlin

p = [0.0f0 1.0f0;
    -2.0f0 0.0f0] # initial geuss at equations

dudt = Chain(
            x->p*x,
            # to start, the network is the identity matrix, so the initial geuss it just the simple ODE alone.
            LinNonlin(Matrix{Float32}(I, 4, 2), convert(Array{Float32,1}, zeros(4)), [0.0f0], tanh),
            LinNonlin(Matrix{Float32}(I, 6, 4), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
            LinNonlin(Matrix{Float32}(I, 6, 6), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
            LinNonlin(Matrix{Float32}(I, 6, 6), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
            LinNonlin(Matrix{Float32}(I, 6, 6), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
            LinNonlin(Matrix{Float32}(I, 6, 6), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
            LinNonlin(Matrix{Float32}(I, 6, 6), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
            LinNonlin(Matrix{Float32}(I, 6, 6), convert(Array{Float32,1}, zeros(6)), [0.0f0], tanh),
            LinNonlin(Matrix{Float32}(I, 3, 6), convert(Array{Float32,1}, zeros(3)), [0.0f0], tanh),
            LinNonlin(Matrix{Float32}(I, 2, 3), convert(Array{Float32,1}, zeros(2)), [0.0f0], tanh)
)

ps = Flux.params(dudt)

n_ode = x->neural_ode(dudt,x,tspan,Tsit5(),saveat=t)#,reltol=1e-7,abstol=1e-9)

pred = n_ode(u0)

predict_n_ode(u=u0) = n_ode(u)

loss_n_ode() = sum(abs2, pendulumData .- predict_n_ode())

cb = function () # callback function to observe training
    display(loss_n_ode())
    # display(("ps",ps))
    # plot current prediction against data
    cur_pred = Flux.data(predict_n_ode())
    pl = plot(t,pendulumData',label="data")
    # plot!(t,cur_pred[1,:]-pendulumData[1,:],label="difference",linestyle=:dot)
    scatter!(pl,t,cur_pred',label="prediction")
    display(plot(pl))
end

cb()

function train(iters::Int = 3; opt = ADAM(0.1), callback = cb, ps = ps, loss_n_ode = loss_n_ode)
    callback()
    data = Iterators.repeated((), iters)
    Flux.train!(loss_n_ode, ps, data, opt, cb = Flux.throttle(callback, 3))
end

train(100,opt=ADAM(0.004))
