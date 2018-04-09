using Revise
using CMAES, Optim
using Base.Test

using BlackBoxOptimizationBenchmarking
const BBOB = BlackBoxOptimizationBenchmarking
f = BBOB.F2

mfit = optimize(
   f,5*rand(3),CMAES.CMA(128,5),
   Optim.Options(iterations=500,store_trace=false,extended_trace=false),
)

Optim.minimum(mfit) < f.f_opt + 1e-6
Optim.minimizer(mfit), f.x_opt[1:3]

# if false

#    using Gadfly
#    fl = layer(z=(x,y) -> f([x;y])-f.f_opt, x=linspace(-5,5,200), y=linspace(-5,5,200), Geom.contour(levels=logspace(-8,8,20)))
#    fl = layer(z=(x,y) -> f([x;y])-f.f_opt, x=linspace(-5,5,100), y=linspace(-5,5,100), Geom.contour())
#    plot(fl)

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

#include(joinpath(Pkg.dir(),"BlackBoxOptimizationBenchmarking","scripts","optimizers_interface.jl"))
import BlackBoxOptimizationBenchmarking: minimizer, minimum, optimize
import Base.string

pinit(D) = 10*rand(D)-5
optimize(opt::Optim.AbstractOptimizer,f,D,run_length) =
Optim.optimize(f, pinit(D), opt, Optim.Options(f_calls_limit=run_length,g_tol=1e-12))

mean_succ, mean_dist, mean_fmin, runtime = BBOB.benchmark([CMAES.CMA(64,5)], [1 2 3 4 5 6 7 8], [20000], 20, 3, 1e-6)

println(mean_succ)
