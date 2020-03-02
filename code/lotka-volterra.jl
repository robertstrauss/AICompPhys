using DifferentialEquations

function lotka_volterra(du,u,p,t)
  x,y = u
  a, b, c, d = p
  du[1] = dx = a*x - b*x*y
  du[2] = dy = -c*y + d*x*y
end

u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5 1.0 3.0 1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p)

sol = solve(prob,Tsit5(),saveat=0.1)
A = sol[1,:]

using Plots
plot(sol)
t = 0:0.1:10.0
scatter!(t,A)

using Flux, DiffEqFlux
diffeq_rd(p,prob,Tsit5(),saveat=0.1)

p = param([2.2, 1.0, 2.0, 0.4]) # Initial Parameter Vector
params = Flux.Params([p])

function predict_rd() # Our 1-layer neural network
  diffeq_rd(p,prob,Tsit5(),saveat=0.1)[1,:]
end

# loss_rd() = sum(abs2,x-1 for x in predict_rd()) # loss function
loss_rd() = sum(abs2,(x=predict_rd(); x[i]-1 for i in 1:size(x)[1])) # loss function

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_rd())
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6),vars=(1,2)))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_rd, params, data, opt, cb = cb)
