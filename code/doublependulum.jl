using Flux, DiffEqFlux, DifferentialEquations, Plots, LinearAlgebra

# initial condition and time span

u0 = [0.985*pi, 0.0]

tspan = (0.0, 20.0)

t = range(tspan[1],tspan[2],length=100)

# pendulum ODE (data to match)

function pendulumODE(du, u, p, t; g=-9.8f0)
    du[1] = u[2]
    du[2] = g*sin(u[1])
end

pendulumProb = ODEProblem(pendulumODE, u0, tspan)

pendulumSolution = solve(pendulumProb, reltol=1E-6)

pendulumData = pendulumSolution(t)

plot(pendulumData')
