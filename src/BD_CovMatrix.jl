
mutable struct BD_CovMatrix{T<:AbstractFloat, K <: MatTypes} <: CovMatrix
    C::K
    B::K
    D::Vector{T}
    c_cov::T
end

function BD_CovMatrix(::Type{K}, n, μ_w) where K <: MatTypes

    c_cov = 1/μ_w * 2/((n+1.3)^2) + (1-1/μ_w)*min(1,2*(μ_w-1)/((n+2)^2 +μ_w)) 
#    c_cov *= 2
    C,B = init_matrix(K,n), init_matrix(K,n)
    D = ones(n)
        
    BD_CovMatrix(C,B,D,c_cov)
end

cov_matrix(cov_mat::BD_CovMatrix) = cov_mat.C
eigvals(cov_mat::BD_CovMatrix) = cov_mat.D

function update_p_σ(::Type{BD_CovMatrix{T,K}},c_σ, p_σ, μ_w, B, ẑ) where {T,K}
    (1-c_σ)*p_σ  + √(c_σ*(2-c_σ)*μ_w)*B*ẑ
end

function update_p_c(::Type{BD_CovMatrix{T,K}},c_c, p_c, H_σ, c_σ, μ_w, BD, ẑ) where {T,K}
    (1-c_c)*p_c + H_σ*√(c_σ*(2 - c_σ)*μ_w)*BD*ẑ
end
    
function update_C(::Type{BD_CovMatrix{T,K}}, C::K, c_cov, μ_w, p_c, w, BD, z, μ) where {T,K}
    C = (1-c_cov) * C +
        c_cov * 1/μ_w * ⊗(K,p_c) +
        c_cov * (1-1/μ_w) * ∑( w[i]* ⊗(K,BD*z[i]) for i=1:μ)
    C
end

function update_state!(d, state::CMAState{T,CM}, method::CMA) where {T,CM <: BD_CovMatrix}
    
    @unpack w,n,μ_w,c_σ,d_σ,c_c,p_σ,p_c,cov_mat,m,chi_D,σ,t,xs,dict_struct = state
    @unpack C,B,D,c_cov = cov_mat
    
    state.f_x_previous = state.f_x
    copy!(state.x_previous, state.x)

    λ = method.λ
    μ = floor(Int,λ/2)

    BD = B*Diagonal(D)

    z = [randn(n) for i=1:λ]
    x = [m + σ*BD*z[i] for i=1:λ]

    fx = eval_objfun(T, λ, d, x, dict_struct)

    idx = sortperm(fx)  
    x,z,fx = x[idx], z[idx], fx[idx]
    
    m = ∑( w[i] * x[i] for i=1:μ )
    ẑ = ∑( w[i] * z[i] for i=1:μ )
    
    p_σ = update_p_σ(CM, c_σ, p_σ, μ_w, B, ẑ)
    H_σ = update_H_σ(CM, p_σ, c_σ, t, n)
    p_c = update_p_c(CM, c_c, p_c, H_σ, c_σ, μ_w, BD, ẑ)
    C = update_C(CM, C, c_cov, μ_w, p_c, w, BD, z, μ)
    
    σ = update_σ(CM, σ, c_σ, d_σ, p_σ, chi_D)
        
    D,B = eig(C)
    B = typeof(C) <: Diagonal ? Diagonal(B) : B
    
    D = sqrt.(abs.(D)) #eigenvalues can become negative because of numerical errors 
        
    t += 1
    state.x = x[1]
    state.f_x = fx[1]
    xs = x

    @pack! cov_mat = C,B,D,c_cov
    @pack! state = w,n,μ_w,c_σ,d_σ,c_c,p_σ,p_c,cov_mat,m,chi_D,σ,t,xs,dict_struct

    false # should the procedure force quit?
end


