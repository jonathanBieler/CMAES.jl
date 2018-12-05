
mutable struct FullCovMatrix{T<:AbstractFloat, K <: MatTypes} <: CovMatrix
    C::K
    c_1::T
    c_μ::T
end

function FullCovMatrix(::Type{K}, n, μ_w) where K <: MatTypes
    c_1 = 2/((n+1.3)^2 + μ_w)
    c_μ = min(1-c_1,2*(μ_w-2+1/μ_w)/((n+2)^2 +μ_w))
    C = init_matrix(K,n)
    
    FullCovMatrix(C,c_1,c_μ)
end

cov_matrix(cov_mat::FullCovMatrix) = cov_mat.C
eigvals(cov_mat::FullCovMatrix) = eig(cov_mat.C)[1] #FIXME maybe a bit costly

mv_rand(n,C) = rand(MultivariateNormal(zeros(n),C))
mv_rand(n,C::Diagonal) = rand(MultivariateNormal(zeros(n),diag(C)))

function update_p_σ(::Type{FullCovMatrix{T,K}},c_σ, p_σ, μ_w, C, Δm, σ) where {T,K}
    (1-c_σ)*p_σ  + √(c_σ*(2-c_σ)*μ_w) * C^(-1/2) * Δm/σ
end

function update_p_c(::Type{FullCovMatrix{T,K}},c_c, p_c, H_σ, c_σ, μ_w, Δm, σ) where {T,K}

#    @show (H_σ*√(c_σ*(2 - c_σ)*μ_w), Δm/σ)
#    (1-c_c/3)*p_c + H_σ*√(c_σ*(2 - c_σ)*μ_w) * 2.5Δm/σ
    (1-c_c)*p_c + H_σ*√(c_σ*(2 - c_σ)*μ_w) * Δm/σ
end

function update_C(::Type{FullCovMatrix{T,K}}, C::K, c_1, c_μ, H_σ, c_c, p_c, w, σ, x, m, μ) where {T,K}

#    c_1 = 0.75*c_1
#    c_µ = 2*c_µ

    C = (1-c_1-c_µ + (1-H_σ)*c_1*c_c*(2-c_c)) * C +
        c_1 * p_c * p_c' +
        c_μ * ∑( w[i]/σ^2*( (x[i]-m)*(x[i]-m)') for i=1:μ)

#    @show (1-c_1-c_µ + (1-H_σ)*c_1*c_c*(2-c_c)) * C
#    @show c_1 * p_c * p_c'
#    @show c_μ * ∑( w[i]/σ^2*( (x[i]-m)*(x[i]-m)') for i=1:μ)
#    @show p_c

    C = (C+C')/2 #keep symmetric part
    C
end
    
#this is probably wrong
function update_C(::Type{FullCovMatrix{T,K}}, C::K, c_1, c_μ, H_σ, c_c, p_c, w, σ, x, m, μ) where {T,K <: Diagonal}

    C = (1-c_1-c_µ + (1-H_σ)*c_1*c_c*(2-c_c)) * C +
        c_1 * Diagonal(p_c.^2) +
        c_μ * ∑( w[i]/σ^2*( Diagonal((x[i]-m).^2) ) for i=1:μ)
    C
end

function update_state!(d, state::CMAState{T,CM}, method::CMA) where {T,CM <: FullCovMatrix}
    
    @unpack w,n,μ_w,c_σ,d_σ,c_c,p_σ,p_c,cov_mat,m,chi_D,σ,t,xs,dict_struct = state
    @unpack C,c_1,c_μ = cov_mat
    
    state.f_x_previous = state.f_x
    copy!(state.x_previous, state.x)

    λ = method.λ
    μ = floor(Int,λ/2)

    x = [m + σ*mv_rand(n,C) for i=1:λ]
    
    fx = eval_objfun(T, λ, d, x, dict_struct)

    idx = sortperm(fx)  
    x, fx = x[idx], fx[idx]
    
    m_t = m
    m = ∑( w[i] * x[i] for i=1:μ)
    
    Δm = m-m_t
    
#    @show C
    
    p_σ = update_p_σ(CM, c_σ, p_σ, μ_w, C, Δm, σ)
    H_σ = update_H_σ(CM, p_σ, c_σ, t, n)
    p_c = update_p_c(CM, c_c, p_c, H_σ, c_σ, μ_w, Δm, σ)
    C = update_C(CM, C, c_1, c_μ, H_σ, c_c, p_c, w, σ, x, m, μ)
    
    σ = update_σ(CM, σ, c_σ, d_σ, p_σ, chi_D)
    
    D = eigvals(cov_mat) 
    
#    @printf("σ: %2.2f H_σ: %2.2f axis-ratio: %2.2e \n",
#            σ, H_σ, maximum(D) / minimum(D) )
            
    t += 1
    state.x = x[1]
    state.f_x = fx[1]
    xs = x

    @pack cov_mat = C,c_1,c_μ
    @pack state = w,n,μ_w,c_σ,d_σ,c_c,p_σ,p_c,cov_mat,m,chi_D,σ,t,xs,dict_struct

    false # should the procedure force quit?
end


