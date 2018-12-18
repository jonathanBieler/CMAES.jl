using CMAES, Optim, LinearAlgebra
using Test

f(x) = sum(abs2.(x))
D = 5

@testset "FullCovMatrix" begin
    
    T = CMAES.FullCovMatrix
    mfit = optimize(
       f,10*rand(D).-5,CMAES.CMA(T,5;dimension=D),
       Optim.Options(f_calls_limit=5000,store_trace=true,extended_trace=true),
    )
    @test Optim.minimum(mfit) < 1e-12
end

@testset "BD_CovMatrix" begin
    
    T = CMAES.BD_CovMatrix
    mfit = optimize(
       f,10*rand(D).-5,CMAES.CMA(T,5;dimension=D),
       Optim.Options(f_calls_limit=5000,store_trace=true,extended_trace=true),
    )
    @test Optim.minimum(mfit) < 1e-12
end

@testset "BD_CovMatrix+Diagonal" begin
    
    mfit = optimize(
       f,10*rand(D).-5,CMAES.CMA{Diagonal,CMAES.BD_CovMatrix}(9,5),
       Optim.Options(f_calls_limit=5000,store_trace=true,extended_trace=true),
    )
    @test Optim.minimum(mfit) < 1e-12
end
