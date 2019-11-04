function pendulum(du,u,p,t)
    x, y = u
    g, r = p
    du[1] = dx = y
    du[2] = dy = g*sin(x)/r
end
#     p[0]*sin(u[0])/p[2] # theta double dot = g*sin(theta)/r



function simpleharm(du,u,p,t)
    x, y = u
    g, r = p
    du[1] = dx = y
    du[2] = dy = g*x/r
end



using DifferentialEquations
u0 = [3.1, 0.0] # initial theta value (rad), initial angular speed
tspan = (0.0, 30.0) # seconds
p = [-9.8, 1.0] # gravity acceleration (m/s/s), pendulum length (m)



prob1 = ODEProblem(pendulum, u0, tspan, p)


prob2 = ODEProblem(simpleharm, u0, tspan, p)


sol1 = solve(prob1,reltol=1e-6)
ss1 = sol1(sol1.t)

sol2 = solve(prob2,reltol=1e-6)
ss2 = sol2(sol1.t)

using Plots



plot(sol2.t, [sol1[1,:], sol2[1,:]], ylim=(-3.2, 3.2))

plot(sol1)
