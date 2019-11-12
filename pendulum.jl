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




u0 = [3.1, 0.0] # initial theta value (rad), initial angular speed
tspan = (0.0, 30.0) # seconds
p = [-9.8, 1.0] # gravity acceleration (m/s/s), pendulum length (m)





using DifferentialEquations


prob1 = ODEProblem(pendulum, u0, tspan, p)

prob2 = ODEProblem(simpleharm, u0, tspan, p)


sol1 = solve(prob1,reltol=1e-6)
ss1 = sol1(sol1.t)

sol2 = solve(prob2,reltol=1e-6)
ss2 = sol2(sol1.t)


using Plots


using Flux, DiffEqFlux

fp = param([-15, 1.0])
fparams = Flux.Params([fp])

predict_rd1() = diffeq_rd(p, prob1, Tsit5(), saveat=0.1)[1,:]
predict_rd2() = diffeq_rd(fp, prob2, Tsit5(), saveat=0.1)[1,:]

loss_rd()     = sum(abs2, x-y for (x, y) in (predict_rd1(), predict_rd2()))

data = Iterators.repeated((), 10)
opt = ADAM(0.1)
function callback()
    display(loss_rd())
    p1sol = solve(prob1,Tsit5(),saveat=0.1)
    p1ss = p1sol(p1sol.t)
    p2sol = solve(remake(prob2,p=Flux.data(fp)),Tsit5(),saveat=0.1)
    p2ss = p2sol(p1sol.t)
    both = [transpose(p1ss), transpose(p2ss)]
    display(plot(both))
    # display(Plots.plot(p1ss, p2ss))
end


callback() # display once

Flux.train!(loss_rd, fparams, data, opt, cb = callback)







using Plots


plot(sol1, ylim=(-10,10))

plot(sol2, ylim=(-10,10))
