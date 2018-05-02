##

using Revise
using CMAES, Optim
using Base.Test
using PDMats

using BlackBoxOptimizationBenchmarking
const BBOB = BlackBoxOptimizationBenchmarking
f = BBOB.F2

include("/Users/jbieler/.julia/v0.6/CMAES/src/CMAES.jl")

mfit = optimize(
   f,5*rand(2),CMAES.CMA(16,5),
   Optim.Options(iterations=20,store_trace=true,extended_trace=true),
)

@assert Optim.minimum(mfit) < f.f_opt + 1e-6
Optim.minimizer(mfit), f.x_opt[1:3]

##

mfit = optimize(
   f,5*rand(2),CMAES.CMA{Diagonal}(116,5),
   Optim.Options(iterations=20,store_trace=true,extended_trace=true),
)

#@assert Optim.minimum(mfit) < f.f_opt + 1e-6
Optim.minimum(mfit) , f.f_opt + 1e-6

x = [mean(mfit.trace[t].metadata["x"])[1] for t=1:length(mfit.trace)]
y = [mean(mfit.trace[t].metadata["x"])[2] for t=1:length(mfit.trace)]
z = [mfit.trace[t].value for t=1:length(mfit.trace)]

plot(y=z-f.f_opt,yintercept=[1e-6],Geom.line,Geom.hline(color=colorant"gray"),Scale.y_log10)

##

    using Gadfly
    fl = layer(z=(x,y) -> f([x;y])-f.f_opt, x=linspace(-5,5,200), y=linspace(-5,5,200), Geom.contour(levels=logspace(-8,8,15)))
#    fl = layer(z=(x,y) -> f([x;y])-f.f_opt, x=linspace(-5,5,100), y=linspace(-5,5,100), Geom.contour())

    x = [mean(mfit.trace[t].metadata["x"])[1] for t=1:length(mfit.trace)]
    y = [mean(mfit.trace[t].metadata["x"])[2] for t=1:length(mfit.trace)]

    plot(fl, 
        layer(x=x,y=y,Geom.line, Theme(default_color=colorant"black") ), 
        layer(x=[f.x_opt[1]],y=[f.x_opt[2]],Geom.point,Theme(default_color=colorant"red")) 
    )
 
##

mfit = optimize(
   f,5*rand(3),CMAES.CMA(3; dimension=3),
   Optim.Options(iterations=500,store_trace=false,extended_trace=false),
)
#

##
 
# if false   
#
#    for t = 2:30#length(mfit.trace)
#        p = mfit.trace[t].metadata["x"]
#        display(plot(
#            [layer(x=[p[1]],y=[p[2]],Geom.point, Theme(highlight_width=0pt)) for p in p]...,
#            Coord.cartesian(xmin=-5,xmax=5,ymin=-5,ymax=5),
#            layer(xintercept=[f.x_opt[1]],yintercept=[f.x_opt[2]],Geom.vline(color=colorant"lightgray"),Geom.hline(color=colorant"lightgray"),),
#            fl,
#        ))
#        sleep(0.01)
#    end

# end

##

#include(joinpath(Pkg.dir(),"BlackBoxOptimizationBenchmarking","scripts","optimizers_interface.jl"))
if false

import BlackBoxOptimizationBenchmarking: minimizer, minimum, optimize
import Base.string

pinit(D) = 10*rand(D)-5
optimize(opt::Optim.AbstractOptimizer,f,D,run_length) =
Optim.optimize(f, pinit(D), opt, Optim.Options(f_calls_limit=run_length,g_tol=1e-12))

mean_succ, mean_dist, mean_fmin, runtime = BBOB.benchmark([CMAES.CMA(64,5)], [1 2 3 4 5], [20000], 20, 3, 1e-6)

println(mean_succ)

end

