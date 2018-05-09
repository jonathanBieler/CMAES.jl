module CMAES
##

    using Optim, Distributions, Parameters, PDMats
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

    struct CMA{K<:MatTypes} <: Optim.ZerothOrderOptimizer
        λ::Int
        σ::Float64
    end

    CMA(λ::Int, σ::Real) = CMA{Matrix}(λ,σ)
    function CMA(σ::Real; dimension=0) 
         @assert dimension > 0 
         CMA{Matrix}(4+floor(Int,3*log(dimension)), σ)
    end
    
    type CMAState{T <: AbstractFloat, K <: MatTypes} <: Optim.ZerothOrderState
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
        c_cov::T
        #
        p_σ::Vector{T}
        p_c::Vector{T}
        C::K
        D::Vector{T}
        B::K
        m::Vector{T}
        chi_D::T
        σ::T
        σ_0::T
        λ::Int
        t::Int
        xs::Vector{Vector{T}}
    end
    
    ∑(x) = sum(x)    
    weights(μ) = [(log(μ+1)-log(i)) for i=1:μ] / ∑( log(μ+1)-log(j) for j=1:μ )
    
    ##

    function init_matrix(method::CMA{K},n) where K <: MatTypes
        K <: Matrix ? diagm(ones(n)) : Diagonal(ones(n))
    end

    function initial_state(method::CMA, options, d, xinit) 
        
        λ = method.λ
        n = length(xinit)
        
        μ = floor(Int,λ/2)
        w = weights(μ)
        
        n, μ_w, c_σ, d_σ, c_c, c_cov = init_constants(method,xinit,λ,w,μ)
        p_σ, p_c = zeros(n), zeros(n)
        C,B = init_matrix(method,n), init_matrix(method,n)
        D = ones(n)
        
        m = xinit
    
        chi_D = √n*(1-1/(4*n) + 1/(21*n^2))
        xs = [zeros(n) for i=1:μ]
        σ_0 = copy(method.σ)
        
        CMAState(
            xinit,xinit,Inf,Inf,w,
            n, μ_w, c_σ, d_σ, c_c, c_cov,
            p_σ, p_c,C,D,B,m,chi_D,method.σ,σ_0,λ,1,xs
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
    
  
    function update_state!(d, state::CMAState{T,K}, method::CMA) where {T,K}
    
        @unpack w,n,μ_w,c_σ,d_σ,c_c,c_cov,p_σ,p_c,C,D,B,m,chi_D,σ,t,xs = state
        
        state.f_x_previous = state.f_x
        copy!(state.x_previous, state.x)

        λ = method.λ
        μ = floor(Int,λ/2)

        BD = B*Diagonal(D)

        z = [randn(n) for i=1:λ]
        x = [m + σ*BD*z[i] for i=1:λ]
    
        fx = zeros(T,λ)
        for i=1:λ
            value!(d,x[i])
            fx[i] = value(d)
            isnan(fx[i]) && error("Function returned NaN for input $(x[i])")
        end

        idx = sortperm(fx)  
        x,z,fx = x[idx], z[idx], fx[idx]
        
        m = ∑( w[i] * x[i] for i=1:μ )
        ẑ = ∑( w[i] * z[i] for i=1:μ )
        
        p_σ = (1-c_σ)*p_σ  + √(c_σ*(2-c_σ)*μ_w)*B*ẑ
        
        H_σ = norm(p_σ) < √(1-(1-c_σ)^(2*(t+1)))*(1.4 + 2/(n+1)) ? 1.0 : 0.0
        
        p_c = (1-c_c)*p_c + H_σ*√(c_σ*(2 - c_σ)*μ_w)*BD*ẑ
           
        C = update_C(C,c_cov, μ_w, p_c, w, BD, z, μ)
        
        σ = σ * exp(c_σ/d_σ*(norm(p_σ)/chi_D -1))
            
#        @show maximum(D)/minimum(D) 
            
        D,B = eig(C)
        B = K <: Diagonal ? Diagonal(B) : B
        
        D = sqrt.(abs.(D)) #eigenvalues can become negative because of numerical errors 
        
        t += 1
        state.x = x[1]
        state.f_x = fx[1]
        xs = x

        @pack state = w,n,μ_w,c_σ,d_σ,c_c,c_cov,p_σ,p_c,C,D,B,m,chi_D,σ,t,xs
                
        false # should the procedure force quit?
    end
    
    ⊗(::Type{Matrix{T}},x)   where T = x * x'
    ⊗(::Type{Diagonal{T}},x) where T = Diagonal(x.^2)
    
    function update_C(C::K, c_cov, μ_w, p_c, w, BD, z, μ) where K <: MatTypes
        C = (1-c_cov) * C +
            c_cov * 1/μ_w * ⊗(K,p_c) +
            c_cov * (1-1/μ_w) * ∑( w[i]* ⊗(K,BD*z[i]) for i=1:μ)
            
        #(C+C')/2
        C
    end
    
    function init_constants(method::CMA{K}, xinit,λ,w,μ) where K <: MatTypes
        n = length(xinit)
    
        μ_w = 1.0 / sum(w.^2)
        c_σ =  (μ_w + 2.0) / (n + μ_w + 5.0)
        
        d_σ = 1.0 + c_σ + 2.0*max(0, √((μ_w-1)/(n+1)) -1)
        
        c_c = (4 + μ_w/n)/(n + 4 + 2μ_w/n) #Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed
        
        c_cov = 1/μ_w * 2/((n+1.3)^2) + (1-1/μ_w)*min(1,2*(μ_w-1)/((n+2)^2 +μ_w)) 
        
#        c_cov *= 2
         
        if K <: Diagonal #A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity
            c_cov = min(1, c_cov*(n+2)/3) # I need to clamp here for low n
        end
        
        n, μ_w, c_σ, d_σ, c_c, c_cov
    end

    function assess_convergence(state::CMAState, d, options)
        
        x_converged, f_converged, g_converged, converged, f_increased = Optim.default_convergence_assessment(state, d, options)
        f_increased = false #disable this one, since error can increase with CMAES

        @unpack n, λ, p_c, σ, σ_0,t,D,C = state

        MaxIter = t > 100 + 50*(n+3)*2/√(λ)

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
        