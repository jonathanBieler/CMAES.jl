module CMAES

##

    using Optim, Distributions, Parameters, PDMats
    import Optim: AbstractObjective, ZerothOrderOptimizer, ZerothOrderState, initial_state, update_state!,
                  trace!, assess_convergence, AbstractOptimizerState, update!, value, value!,
                  pick_best_x, pick_best_f, f_abschange, x_abschange

    #diagonal doesn't work on v0.6, but might in 0.7
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
    
    type CMAState{T, K <: MatTypes} <: Optim.ZerothOrderState
        x::Array{T,1}
        x_previous::Array{T,1} #this is used by default convergence test
        f_x::T
        f_x_previous::T
        #
        w::Vector{T}
        D::Int
        μ_w::T
        c_σ::T
        d_σ::T
        c_c::T
        c_1::T
        c_μ::T
        #
        p_σ::Vector{T}
        p_c::Vector{T}
        C::K
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

    function init_C(method::CMA{K},D) where K <: MatTypes
        K <: Matrix ? diagm(ones(D)) : Diagonal(ones(D))
    end

    function initial_state(method::CMA, options, d, xinit) 
        
        λ = method.λ
        D = length(xinit)
        
        μ = floor(Int,λ/2)
        w = weights(μ)
        
        D, μ_w, c_σ, d_σ, c_c, c_1, c_μ = init_constants(method,xinit,λ,w,μ)
        p_σ, p_c = zeros(D), zeros(D)
        C = init_C(method,D)
        
        m = xinit
    
        chi_D = √D*(1-1/(4*D) + 1/(21*D^2))
        xs = [zeros(D) for i=1:μ]
        σ_0 = copy(method.σ)
        
        CMAState(
            xinit,xinit,Inf,Inf,w,
            D, μ_w, c_σ, d_σ, c_c, c_1, c_μ,
            p_σ, p_c,C,m,chi_D,method.σ,σ_0,λ,1,xs
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
    
    mv_rand(D,C) = rand(MultivariateNormal(zeros(D),C))
    mv_rand(D,C::Diagonal) = rand(MultivariateNormal(zeros(D),diag(C)))

    function update_state!{T}(d, state::CMAState{T}, method::CMA)
    
        @unpack w,D,μ_w,c_σ,d_σ,c_c,c_1,c_μ,p_σ,p_c,C,m,chi_D,σ,t,xs = state
        
        state.f_x_previous = state.f_x
        copy!(state.x_previous, state.x)

        λ = method.λ
        μ = floor(Int,λ/2)

        x = [m + σ*mv_rand(D,C) for i=1:λ]
    
        fx = zeros(T,λ)
        for i=1:λ
            value!(d,x[i])
            fx[i] = value(d)
            isnan(fx[i]) && error("Function return NaN for input $(x[i])")
        end

        idx = sortperm(fx)  
        x, fx = x[idx], fx[idx]
        
        m_t = m
        m = ∑( w[i] * x[i] for i=1:μ)
        
        Δm = m-m_t
        
        p_σ = update_p_σ(c_σ, p_σ, μ_w, C, Δm, σ)
        
        h_σ = norm(p_σ) < √(1-(1-c_σ)^(2*(t+1)))*(1.4 + 2/(D+1)) ? 1.0 : 0.0
        
        p_c = (1-c_c)*p_c + h_σ*√(c_σ*(2 - c_σ)*μ_w) * Δm/σ
           
        C = update_C(C, c_1, c_μ, h_σ, c_c, p_c, w, σ, x, m, μ)
        
        σ = σ * exp(c_σ/d_σ*(norm(p_σ)/chi_D -1))
            
        @show p_σ
            
        t += 1
            
        state.x = x[1]
        state.f_x = fx[1]
        xs = x

        @pack state = w,D,μ_w,c_σ,d_σ,c_c,c_1,c_μ,p_σ,p_c,C,m,chi_D,σ,t,xs
                
        false # should the procedure force quit?
    end
    
    function update_C(C::Matrix, c_1, c_μ, h_σ, c_c, p_c, w, σ, x, m, μ)
    
        C = (1-c_1-c_µ + (1-h_σ)*c_1*c_c*(2-c_c)) * C +
            c_1 * p_c * p_c' +
            c_μ * ∑( w[i]/σ^2*( (x[i]-m)*(x[i]-m)') for i=1:μ)
            
        C = (C+C')/2 #keep symmetric part
        C
    end
        
    #this is probably wrong
    function update_C(C::Diagonal, c_1, c_μ, h_σ, c_c, p_c, w, σ, x, m, μ)
    
        C = (1-c_1-c_µ + (1-h_σ)*c_1*c_c*(2-c_c)) * C +
            c_1 * Diagonal(p_c.^2) +
            c_μ * ∑( w[i]/σ^2*( Diagonal((x[i]-m).^2) ) for i=1:μ)
            
        C
    end
    
    function update_p_σ(c_σ, p_σ, μ_w, C, Δm, σ)
        (1-c_σ)*p_σ  + √(c_σ*(2-c_σ)*μ_w) * C^(-1/2) * Δm/σ
    end
    
    function update_p_σ(c_σ, p_σ, μ_w, C::Diagonal, Δm, σ)
        (1-c_σ)*p_σ  + √(c_σ*(2-c_σ)*μ_w) * C^(-1/2) * Δm/σ
    end
    
    function assess_convergence(state::CMAState, d, options)
        
        x_converged, f_converged, g_converged, converged, f_increased = Optim.default_convergence_assessment(state, d, options)
        f_increased = false #disable this one, since error can increase with CMAES

        @unpack D, λ, p_c, σ, σ_0,t,C = state

        d,_ = eig(C) #FIXME maybe a bit costly

        MaxIter = t > 100 + 50*(D+3)*2/√(λ)

        TolX = t > 1 && all(p_c * σ/σ_0 .< 1e-16) && all(sqrt.(diag(C)) * σ/σ_0  .< 1e-16)
        TolUpSigma = σ/σ_0 > 10^20 * √(maximum(d))
        ConditionCov = abs(maximum(d)) / abs(minimum(d)) > 10^14
        
        converged = MaxIter || TolX || TolUpSigma || ConditionCov

        x_converged, f_converged, g_converged, converged, f_increased
    end

    f_abschange(d::AbstractObjective, state::CMAState) = f_abschange(state.f_x, state.f_x_previous)
    x_abschange(state::CMAState) = x_abschange(state.x, state.x_previous)

    pick_best_f(f_increased, state::CMAState, d) = state.f_x 
    pick_best_x(f_increased, state::CMAState) = state.x

    function init_constants(method::CMA{K}, xinit,λ,w,μ) where K <: MatTypes
        D = length(xinit)
    
        μ_w = 1.0 / sum(w.^2)
        c_σ =  (μ_w + 2.0) / (D + μ_w + 5.0)
        
        d_σ = 1.0 + c_σ + 2.0*max(0, √((μ_w-1)/(D+1)) -1) 
        
        c_c = (4 + μ_w/D)/(D + 4 + 2μ_w/D) 
        
        c_1 = 2/((D+1.3)^2 + μ_w)
        c_μ = min(1-c_1,2*(μ_w-2+1/μ_w)/((D+2)^2 +μ_w))
        
        if K == Diagonal #A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity
#            d_σ *= 100#1.0 + c_σ + 2.0*max(0, √((μ_w-1)/(D+1)) -1) 
#            
#            c_1 = (D+2)/3 * 1/μ_w * 2/(√2 + D)^2
#            c_μ = (D+2)/3 * (1-1/μ_w)*min(1,(2μ_w-1)/((D+2)^2 +μ_w))
        end

        D, μ_w, c_σ, d_σ, c_c, c_1, c_μ
    end
   
##
end

##





##
        