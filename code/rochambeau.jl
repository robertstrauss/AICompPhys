using DifferentialEquations
function rochambeau(du,u,p,t)
  x,y,z = u
  a, b, c, d, e, f, g, h, i = p
  du[1] = dx = a*x - b*x*y + c*x*z #a*x - b*x*y
  du[2] = dy = d*y - e*y*z + f*y*x
  du[3] = dz = g*z - h*z*x + i*z*y
end
u0 = [1.0 1.0 1.0]
tspan = (0.0, 20.0)
p = [0.1 1.9 1.5 -0.5 2.1 1.2 1.0 1.5 1.2]
prob = ODEProblem(rochambeau,u0,tspan,p)

sol = solve(prob)
using Plots
plot(sol)
