module CMAES
##

    using Optim, Distributions, Parameters, PDMats
    using LinearAlgebra

    import Optim: AbstractObjective, ZerothOrderOptimizer, ZerothOrderState, initial_state, update_state!,
                  trace!, assess_convergence, AbstractOptimizerState, update!, value, value!,
                  pick_best_x, pick_best_f, f_abschange, x_abschange

    const MatTypes = Union{Matrix,Diagonal}
    
    @static if VERSION < v"0.7.0-DEV.3510"
        import Base.LinAlg: copy_oftype, checksquare
    else
        using LinearAlgebra
        import LinearAlgebra: copy_oftype, checksquare
    end
    
    import Base.(^)
    function (^)(A::Diagonal{T}, p::Real) where T
    
        n = checksquare(A)
        TT = Base.promote_op(^, T, typeof(p))
        retmat = copy_oftype(A, TT)
        for i in 1:n
            retmat[i, i] = retmat[i, i] ^ p
        end
        return retmat
    end

    abstract type CovMatrix end
        
    struct CMA{K<:MatTypes, CM <: CovMatrix} <: Optim.ZerothOrderOptimizer
        λ::Int
        σ::Float64
    end
    
    init_matrix(::Type{Matrix},n) = Matrix(Diagonal(ones(n)))
    init_matrix(::Type{Diagonal},n) = Diagonal(ones(n))
        
    mutable struct CMAState{T <: AbstractFloat, CM <: CovMatrix} <: Optim.ZerothOrderState
        x::Array{T,1}
        x_previous::Array{T,1} #this is used by default convergence test
        f_x::T
        f_x_previous::T
        #
        w::Vector{T}
        n::Int
        μ_w::T
        c_σ::T
        d_σ::T
        c_c::T
        #
        p_σ::Vector{T}
        p_c::Vector{T}
        cov_mat::CM
        m::Vector{T}
        chi_D::T
        σ::T
        σ_0::T
        λ::Int
        t::Int
        xs::Vector{Vector{T}}
    end
    
    ∑(x) = sum(x)    
    weights(μ) = log(μ + 1/2) .- log.(1:μ) #[(log(μ+1)-log(i)) for i=1:μ] / ∑( log(μ+1)-log(j) for j=1:μ )
    
    include("FullCovMatrix.jl")
    include("BD_CovMatrix.jl")
    
    CMA(λ::Int, σ::Real) = CMA{Matrix}(λ,σ)
    function CMA(σ::Real; dimension=0) 
         @assert dimension > 0 
         CMA{Matrix,BD_CovMatrix}(4+floor(Int,3*log(dimension)), σ)
    end
    
    function CMA(::Type{T},σ::Real; dimension=0) where T <: CovMatrix
         @assert dimension > 0 
         CMA{Matrix,T}(4+floor(Int,3*log(dimension)), σ)
    end
    
    ## Dict support (view previous commit)
    
    npara(xinit::Vector) = length(xinit)

    ##

    function initial_state(method::CMA{K,CM}, options, d, xinit) where {K <: MatTypes, CM <: CovMatrix}
        
        λ = method.λ
        n = npara(xinit)
        
        μ = floor(Int,λ/2)
        w = weights(μ)
        normalize!(w, 1)
        
        m = xinit
        n, μ_w, c_σ, d_σ, c_c = init_constants(method,n,λ,w,μ)
        p_σ, p_c = zeros(n), ones(n)
        cov_mat = CM(K, n, μ_w)
        
        chi_D = √n*(1-1/(4n) + 1/(21n^2))
        xs = [zeros(n) for i=1:μ]
        σ_0 = copy(method.σ)
        
        CMAState(
            xinit,xinit,Inf,Inf,w,
            n, μ_w, c_σ, d_σ, c_c,
            p_σ, p_c,cov_mat,m,chi_D,method.σ,σ_0,λ,1,xs,
        )
    end
    
    function trace!(tr, d, state, iteration, method::CMA, options)
        dt = Dict()
        if options.extended_trace
            dt["x"] = state.xs
        end
        g_norm = 0.0
        update!(tr,
        iteration,
        state.f_x,
        g_norm,
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
        options.callback)
    end
    
  
    function eval_objfun(T, λ, d, x)
        fx = zeros(T,λ)
        for i=1:λ
            value!(d,x[i])
            fx[i] = value(d)
            isnan(fx[i]) && error("Function returned NaN for input $(x[i])")
        end
        fx
    end
    
    function update_H_σ(::Type{CM},p_σ, c_σ, t, n) where CM <: CovMatrix
        norm(p_σ) < √(1-(1-c_σ)^(2*(t+1)))*(1.4 + 2/(n+1)) ? 1.0 : 0.0
    end
    
    function update_σ(::Type{CM},σ, c_σ, d_σ, p_σ, chi_D) where CM <: CovMatrix
    σ * exp(c_σ/d_σ*(norm(p_σ)/chi_D -1))
    end
    
    ⊗(::Type{Matrix{T}},x)   where T = x * x'
    ⊗(::Type{Diagonal{T,Vector{T}}},x) where T = Diagonal(x.^2)
    
    function init_constants(method::CMA{K},n,λ,w,μ) where K <: MatTypes
        
        μ_w = 1 / sum(w.^2)
        c_σ =  (μ_w + 2) / (n + μ_w + 5)
         
        d_σ = 1 + c_σ + 2*max(0, √((μ_w-1)/(n+1)) -1)
        
        c_c = (4 + μ_w/n)/(n + 4 + 2μ_w/n) #Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed
#        c_c = 4 / (n + 4)
         
#        if K <: Diagonal #A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity
#            c_cov = min(1, c_cov*(n+2)/3) # I need to clamp here for low n
#        end
        
        n, μ_w, c_σ, d_σ, c_c
    end

#(w, μeff, cc, cσ, c1, cμ, dσ) = ([0.637043, 0.28457, 0.0783872], 2.0286114646100617, 0.5714285714285714, 0.5017818438926943, 0.09747248265066792, 0.038593139193450914, 1.5017818438926942)([0.637043, 0.28457, 0.0783872], 2.0286114646100617, 0.5714285714285714, 0.5017818438926943, 0.09747248265066792, 0.038593139193450914, 1.5017818438926942)

    function assess_convergence(state::CMAState, d, options)
        
        x_converged, f_converged, g_converged, converged, f_increased = Optim.default_convergence_assessment(state, d, options)
        f_increased = false #disable this one, since error can increase with CMAES

        @unpack n, λ, p_c, σ, σ_0,t, cov_mat = state
        C,D = cov_matrix(cov_mat), eigvals(cov_mat)

        MaxIter = false# t > 100 + 50*(n+3)*2/√(λ)

        TolX = t > 1 && all(p_c * σ/σ_0 .< 1e-16) && all(sqrt.(diag(C)) * σ/σ_0  .< 1e-16)
        TolUpSigma = σ/σ_0 > 10^20 * √(maximum(D))
        ConditionCov = abs(maximum(D)) / abs(minimum(D)) > 10^14
        
        converged = MaxIter || TolX || TolUpSigma || ConditionCov

        x_converged, f_converged, g_converged, converged, f_increased
    end

    f_abschange(d::AbstractObjective, state::CMAState) = f_abschange(state.f_x, state.f_x_previous)
    x_abschange(state::CMAState) = x_abschange(state.x, state.x_previous)

    pick_best_f(f_increased, state::CMAState, d) = state.f_x 
    pick_best_x(f_increased, state::CMAState) = state.x
   
##
end

##





##
        