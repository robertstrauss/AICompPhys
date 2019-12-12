using DifferentialEquations
using Plots
using Flux, DiffEqFlux


function pendulum(du,u,p,t)
    x, y = u
    g, r = p
    du[1] = dx = y
    du[2] = dy = g*sin(x)/r
end



function simpleharm(du,u,p,t)
    x, y = u
    g, r = p
    du[1] = dx = y
    du[2] = dy = g*x/r
end




u0 = [3.0, 0.0] # initial theta value (rad), initial angular speed
tspan = (0.0, 30.0) # seconds
p = [-9.8, 3.0] # gravity acceleration (m/s/s), pendulum length (m)



prob1 = ODEProblem(pendulum, u0, tspan, p)

prob2 = ODEProblem(simpleharm, u0, tspan, p)


sol1 = solve(prob1,reltol=1e-6)
ss1 = sol1(sol1.t)

sol2 = solve(prob2,reltol=1e-6)
ss2 = sol2(sol1.t)

plot(ss1)
plot(ss2)








u0 = Float32[2.; 0.]
datasize = 50
tspan = (0.0f0,10.0f0)

t = range(tspan[1],tspan[2],length=datasize)
ode_data = Array(solve(prob1,Tsit5(),saveat=t))



dudt = Chain(x -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))
ps = Flux.params(dudt)
n_ode = x->neural_ode(dudt,x,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)



pred = n_ode(u0) # Get the prediction using the correct initial condition
scatter(t,ode_data[1,:],label="data")
scatter!(t,Flux.data(pred[1,:]),label="prediction")



function predict_n_ode()
  n_ode(u0)
end
loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())



data = Iterators.repeated((), 20)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_n_ode())
  # plot current prediction against data
  cur_pred = Flux.data(predict_n_ode())
  pl = scatter(t,ode_data[1,:],label="data")
  scatter!(pl,t,cur_pred[1,:],label="prediction")
  display(plot(pl))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)





0







fp = param([-1.0, 1.0]) # Initial Parameter Vector
fparams = Flux.Params([fp])

diffeq_rd(fp, prob2, Tsit5(), saveat=0.1)
predict_rd() = diffeq_rd(fp, prob2, Tsit5(), saveat=0.1)[1,:]


loss_rd()     = sum(x^2 for x in predict_rd())

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
callback = function()
    display(loss_rd())
    # using `remake` to re-create our `prob` with current parameters `p`
    display(plot(solve(remake(prob2,p=Flux.data(fp)),Tsit5(),saveat=0.1)))
    # display(Plots.plot(p1ss, p2ss))
end


callback() # display once

Flux.train!(loss_rd, fparams, data, opt, cb = callback)








using Plots

plot(sol1, ylim=(-10,10))

plot(sol2, ylim=(-10,10))
