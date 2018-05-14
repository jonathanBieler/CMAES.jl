##

using Gadfly
using CMAES, Optim
using Base.Test
using PDMats

using BlackBoxOptimizationBenchmarking
const BBOB = BlackBoxOptimizationBenchmarking
f = enumerate(BBOBFunction)[5]

include("/Users/jbieler/.julia/v0.6/CMAES/src/CMAES.jl")

T = CMAES.BD_CovMatrix

mfit = optimize(
   f,10*rand(3)-5,CMAES.CMA(T,3;dimension=3),
   Optim.Options(f_calls_limit=2000,store_trace=true,extended_trace=true),
)

z = [mfit.trace[t].value for t=1:length(mfit.trace)]
display(plot(y=z-f.f_opt+1e-16,yintercept=[1e-6],Geom.line,Geom.hline(color=colorant"gray"),Scale.y_log10))

#@assert Optim.minimum(mfit) < f.f_opt + 1e-6
#Optim.minimizer(mfit), f.x_opt[1:3]

##

es = cma.CMAEvolutionStrategy(pinit(3), 1, Dict("verb_log"=>0,"verb_disp"=>1,"maxfevals"=>1000))
mfit = es[:optimize](f)
mfit[:result]

##

using CMAES2

xmin, fmin, = CMAES2.minimize(f.f, 10*rand(3)-5, 3, fill(-100,3), fill(100,3); maxfevals = 2000)

@assert fmin < f.f_opt + 1e-6

##

using BlackBoxOptimizationBenchmarking
const BBOB = BlackBoxOptimizationBenchmarking

BBOB.benchmark(BBOB.OptFun(CMAES.CMA(12,10.0),4), 5_000, 15, 3, 1e-6) 

## Diagonal one

mfit = optimize(
   f,[-5.;4.],CMAES.CMA{Diagonal}(4,1),
   Optim.Options(iterations=2000,store_trace=true,extended_trace=true),
)

#@assert Optim.minimum(mfit) < f.f_opt + 1e-6
Optim.minimum(mfit) , f.f_opt + 1e-6

x = [mean(mfit.trace[t].metadata["x"])[1] for t=1:length(mfit.trace)]
y = [mean(mfit.trace[t].metadata["x"])[2] for t=1:length(mfit.trace)]
z = [mfit.trace[t].value for t=1:length(mfit.trace)]

plot(y=z-f.f_opt+1e-16,yintercept=[1e-6],Geom.line,Geom.hline(color=colorant"gray"),Scale.y_log10)

##

    using Gadfly
    fl = layer(z=(x,y) -> f([x;y])-f.f_opt, x=linspace(-6,6,400), y=linspace(-6,6,400), Geom.contour(levels=logspace(-4,8,20)))
#    fl = layer(z=(x,y) -> f([x;y])-f.f_opt, x=linspace(-5,5,100), y=linspace(-5,5,100), Geom.contour())

    x = [mean(mfit.trace[t].metadata["x"])[1] for t=1:length(mfit.trace)]
    y = [mean(mfit.trace[t].metadata["x"])[2] for t=1:length(mfit.trace)]

    plot(
        layer(x=[f.x_opt[1]],y=[f.x_opt[2]],Geom.point,Theme(default_color=colorant"red")), 
        layer(x=x,y=y,Geom.line(preserve_order=true), Theme(default_color=colorant"black") ),
        fl, 
        Scale.color_continuous(colormap=p->Colors.RGB(p^0.4,p/2+0.4,p/3+0.4)),
        Coord.cartesian(xmin=-6,xmax=6,ymin=-5,ymax=6),
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

import BlackBoxOptimizationBenchmarking: minimizer, minimum, optimize
import Base.string

pinit(D) = 10*rand(D)-5
optimize(opt::Optim.AbstractOptimizer,f,D,run_length) =
Optim.optimize(f, pinit(D), opt, Optim.Options(f_calls_limit=run_length,g_tol=1e-12))

mean_succ, mean_dist, mean_fmin, runtime = BBOB.benchmark([CMAES.CMA(16,5)], [1 2 3 4 5], [20000], 20, 3, 1e-6)

println(mean_succ)



##

