using Gadfly

##

run_bench = true
addprocs(1)

## Setup

@everywhere begin

    import Base: minimum, string
    
    using BlackBoxOptimizationBenchmarking
    const BBOB = BlackBoxOptimizationBenchmarking

    include("/Users/jbieler/.julia/v0.6/CMAES/src/CMAES.jl")
    include("/Users/jbieler/.julia/v0.6/CMAES2/src/CMAES2.jl")
    include(joinpath(Pkg.dir(),"BlackBoxOptimizationBenchmarking/scripts/optimizers_interface.jl"))

    function optimize(::Type{CMAES2.CMAESOpt},f,D,run_length) 
        try rm("CMAES.jld") end
        (CMAES2_res(), CMAES2.minimize(f, pinit(D), 5, fill(-100,D), fill(100,D); maxfevals = run_length)...)
    end
    
    struct CMAES2_res end
           
    xmin(mfit::Tuple{CMAES2_res,Array{Float64,1},Float64,Int64}) = mfit[3]
    minimizer(mfit::Tuple{CMAES2_res,Array{Float64,1},Float64,Int64}) = mfit[2]
   
    dimensions = [3]
    Ntrials = 3
    Δf = 1e-6
#    run_lengths = round.(Int,linspace(20,60_000,20))
    run_lengths = round.(Int,linspace(20,6_000,6))
    funcs = 1:6#length(enumerate(BBOBFunction))
    
end

string(opt::CMAES.CMA{T,K}) where {T,K} = string(typeof(opt).name,K)

optimizers = [
    CMAES.CMA(CMAES.FullCovMatrix,5.0,;dimension=3),
#    CMAES.CMA(CMAES.BD_CovMatrix,5.0,;dimension=3),
#    CMAES2.CMAESOpt,
#    CMAES.CMA(12,10.0),
#    PyCMA(),
]

opt_strings = map(string,optimizers)

## run benchmark

mean_succ, mean_dist, mean_fmin, runtime = BBOB.benchmark(
    optimizers, funcs, run_lengths, Ntrials, dimensions, Δf,
)

## make plots

outdir = joinpath(Pkg.dir(),"CMAES","plots")
mkpath(outdir)
cols = Colors.distinguishable_colors(size(mean_succ,1)+1,colorant"white")[2:end]

#all together

idx = sortperm([mean(mean_succ[i,:,end,:]) for i=1:length(optimizers)],rev=true)

p = plot(
    [layer(
        x=collect(run_lengths),y=mean(mean_succ[i,:,:,:],(1,3)),Geom.line,
        Theme(default_color=cols[i], line_style= mod(i,2)==0 ? :solid : :dash, line_width=2pt)
    ) for i in 1:size(mean_succ,1)]...,
    Coord.cartesian(ymax=1),
    Guide.title("All functions"), Guide.xlabel("Run Length"), Guide.ylabel("Success rate"),
    Guide.manual_color_key("", opt_strings[idx], cols[idx])
)


#draw(PNG(joinpath(outdir,"mean_succ.png"),dpi=150,16cm,10cm),p)

##
# each dimension
for (k,D) in zip(1:length(dimensions), dimensions)

    idx = sortperm([mean(mean_succ[i,:,end,k]) for i=1:length(optimizers)],rev=true)

    p = plot(
        [layer(
            x=collect(run_lengths),y=mean(mean_succ[i,:,:,k],1),Geom.line,
            Theme(default_color=cols[i], line_style= mod(i,2)==0 ? :solid : :dash, line_width=2pt)
        ) for i in 1:size(mean_succ,1)]...,
        Coord.cartesian(ymax=1),
        Guide.title("All functions, D: $D"), Guide.xlabel("Run Length"), Guide.ylabel("Success rate"), 
        Guide.manual_color_key("", opt_strings[idx], cols[idx])
    )
#    draw(PNG(joinpath(outdir,"mean_succ_$(D).png"),dpi=150,16cm,10cm),p)

end


## runtime

idx = sortperm(mean(runtime,2)[:],rev=true)

p = plot(
    layer(
        y=opt_strings[idx], x=(mean(runtime,2)/Base.minimum(mean(runtime,2)))[idx],
        color = cols[idx],
        Geom.bar(orientation=:horizontal)
    ),
    Guide.title("All functions"), Guide.ylabel(""), 
    Guide.xlabel("Relative Run Time",orientation=:horizontal), 
    Scale.x_log10,
#    Scale.y_log10,
)

#draw(PNG(joinpath(outdir,"runtime.png"),dpi=150,16cm,10cm),p)
p

##
