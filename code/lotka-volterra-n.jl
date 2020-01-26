using DifferentialEquations, Flux, Plots, DiffEqFlux, LinearAlgebra

eye(size) = Matrix{Float32}(I, size)

struct fluxor
    dudt
    ps
    truth
    n_ode::Function
    predict::Function
    loss::Function
    cb::Function
    train::Function
end
function fluxor(dudt, truth)
    fl = fluxor(dudt)
    loss() = sum(abs2, truth .- fl.predict())
    return fluxor(fl.dudt, fl.ps, truth, fl.n_ode, fl.predict, loss, fl.cb, fl.train)
end
function fluxor(dudt)
    ps = Flux.params(dudt)
    n_ode(x) = neural_ode(dudt,x,tspan,Tsit5(),saveat=t)#,reltol=1e-7,abstol=1e-9)
    predict(u=u0) = n_ode(u)
    truth = n_ode(u0)
    loss() = sum(abs2, truth .- predict())
    cb = function () # callback function to observe training
        display(loss())
        # plot current prediction against data
        cur_pred = Flux.data(predict())
        try
            pl = plot(t,truth',label="truth")
            scatter!(pl,t,cur_pred',label="prediction")
            display(plot(pl))
        catch
            println("what?!?")
        end
    end
    # cb()
    function train(iters::Int = 3; opt = ADAM(0.1), callback = cb, ps = ps, loss_n_ode = loss, throttle=1)
        Flux.train!(loss_n_ode, ps, Iterators.repeated((), iters), opt, cb = Flux.throttle(callback, throttle))
    end
    return fluxor(dudt, ps, truth, n_ode, predict, loss, cb, train)
end



struct LVn
    A
    r
    LVn(A, r) = new(Flux.param(A), Flux.param(r))
end
LVn(abcd) = (
    s = size(abcd)[1];
    println(abcd);
    r = collect([abcd[i,i] for i in 1:s]);
    A = abcd .- r.*eye((s,s));
    LVn(A, r)
)
(m::LVn)(x) = x .* ( m.r + m.A*x )
Flux.@treelike LVn
# function lotka_volterra(du,u,p,t)
#   x,y = u
#   du[1] = dx =    p[1,1]*x + p[1,2]*x*y
#   du[2] = dy =  p[2,1]*x*y +   p[2,2]*y
# end
#
# prob = ODEProblem(lotka_volterra,u0,tspan,p)
# sol = solve(prob,Tsit5(),saveat=t)
# data = Array(sol)
#
# plot(data')

#=
    for three species x, y, z (a..i are constant parameters)
        dx/dt =  ax + bxy + cxz
        dy/dt = dyx +  ey + fyz
        dz/dt = gzx + hzy +  iz
    rewritten with matricies and vectors
        d/dt [x; y; z] = [ ax + bxy + cxz;
                          dyx +  ey + fyz;
                          gzx + hzy +  iz]
    the right can be written as the element-wise product of the population vector [x; y; z] on another vector
        d/dt [x; y; z] = [x; y; z] ∘ [a  + by + cz;
                                     d*x + e  + fz;
                                      gx + hy + i]

        d/dt [x; y; z] = [x; y; z] ∘ ( [a; e; i] + [0  + by + cz;
                                                   d*x + 0  + fz;
                                                    gx + hy + 0] )
    the vector of sums is the product of a vector of constancts and the population vecto [x; y; z]
        d/dt [x; y; z] = [x; y; z] ∘ ( [a; e; i] + [0  b  c;
                                                    d  0  f;
                                                    g  h  0]*[x; y; z] )
    let p be the population vector [x; y; z],
    r be the vector [a; e; i] and A be the matrix [0 b c; d 0 f; g h 0]
        dp/dt = p ∘ (r + A*p)

    this can be extended to any number of dimensions n as long as
        A is an nxn matrix with zeroes in the diagonal
        r is an n-long vecor
        p is an n-long vector
    the constant parameters in A and r define the interactions of the species
=#


u0 = Float32[1.0, 1.0]
tspan = (0.0,50.0)
t = range(tspan[1], tspan[2], length=100)
p = Float32[1.5 -1.0; 1.0 -3.0]





lvT = LVn(p.+Float32[0.01])
dudtT = Chain(lvT)
#fl = fluxor(dudt)
#data = Array(Flux.data(fl.predict()))
#plot(data')


n_odeT(x) = neural_ode(dudtT,x,tspan,Tsit5(),saveat=t)#,reltol=1e-7,abstol=1e-9)
truth = Flux.data(n_odeT(u0)) #must not be tracked

lv = LVn(p)
dudt = Chain(lv)
ps = Flux.params(dudt)
n_ode(x) = neural_ode(dudt,x,tspan,Tsit5(),saveat=t)#,reltol=1e-7,abstol=1e-9)
predict(u=u0) = n_ode(u)


loss() = sum(abs2, truth .- predict())
cb = function () # callback function to observe training
    display(loss())
    # plot current prediction against data
    cur_pred = Flux.data(predict())
    try
        pl = plot(t,truth',label="truth")
        scatter!(pl,t,cur_pred',label="prediction")
        display(plot(pl))
    catch
        println("what?!?")
    end
end

cb()
dta = Iterators.repeated((),3)
opt = ADAM(0.01)
Flux.train!(loss, ps, dta,  opt, cb = cb)
#Flux.train!(loss, ps, dta,  opt=ADAM(0.01), cb = cb)




u0T = Float32[2.0, 2.0, 2.0]
lv3T = LVn(Float32[0.5 -0.8 -1.1; 0.9 0.5 -1.3; 0.45 1.0 -0.7])
lv3T = LVn(Float32[0.9 -0.8 -1.1; 0.9 0.5 -1.3; 0.45 1.0 -0.7])
dudt3T = Chain(lv3T)
#fl3T = fluxor(dudt3T)
n_odeT(x) = neural_ode(dudt3T,x,tspan,Tsit5(),saveat=t)#,reltol=1e-7,abstol=1e-9)
truth  = Flux.data(n_odeT(u0T))


plot(truth')

truth = truth[2:3,:]
scatter!(truth')

u0=Float32[2.0,2.0]
p = Float32[  0.5 -1.3;  1.0 -0.7]
lv = LVn(p)
# dudt = Chain(Dense(2,16,leakyrelu),Dense(16,32,swish),Dense(32,16,leakyrelu),Dense(16,8,swish),Dense(8,8,tanh),Dense(8,2))
dudt = Chain(lv,
   SkipConnection(Chain(Dense(2,16,leakyrelu),Dense(16,16,tanh),Dense(16,16,swish),Dense(16,16,tanh),Dense(16,2)),(x,y)-> y+0.01*x.*y))
# dudt = Chain(lv)
# dudt = Chain(lv,Dense(2,2))
ps = Flux.params(dudt)
n_ode(x) = neural_ode(dudt,x,tspan,Tsit5(),saveat=t)#,reltol=1e-7,abstol=1e-9)
predict(u=u0) = n_ode(u)


loss() = sum(abs2, truth .- predict())
cb = function () # callback function to observe training
    display(loss())
    # plot current prediction against data
    cur_pred = Flux.data(predict())
    try
        pl = plot(t,truth',label="truth")
        scatter!(pl,t,cur_pred',label="prediction")
        display(plot(pl))
    catch
        println("what?!?")
    end
end

cb()
dta = Iterators.repeated((),30)
opt = ADAM(0.008)
Flux.train!(loss, ps, dta,  opt, cb = cb)
opt = ADAM(0.0005)
dta = Iterators.repeated((),300)

Flux.train!(loss, ps, dta,  opt, cb = cb)
#Flux.train!(loss, ps, dta,  opt=ADAM(0.01), cb = cb)
dta = Iterators.repeated((),10)
opt = ADAM(0.05)
Flux.train!(loss, ps, dta,  opt, cb = cb)





u0 = Float32[1.0, 1.0]
lv3nn = LVn(p) # 2d params
dudt3nn = Chain(lv3nn)
fl3nn = fluxor(dudt3nn, data3[1:2,:])
data3nn = Array(Flux.data(fl3nn.predict()))
plot(data3nn')

fl3nn.train(20,opt=ADAM(0.01),throttle=3)

#
