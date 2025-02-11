struct ArNet
    H::Array{Array{Float64,1},1}
    G::Array{Array{Float64,2},1}
    J::Array{Array{Float64, 3}}
    f1::Array{Float64,1}
end

struct ArVar{T, Ti}
    q::Int
    L::Int
    M::Int
    d::Int
    msa::Array{Ti,2}
    weights::Array{T,1}
    Y::Array{T,2}
    lambdaH::T
    lambdaJ::T
    lambdaG::T
    δ::Array{Int,3}  # Fix δ type to match initialization
    f1::Array{T,1} #frequency count of first position -> works as field for position 1

    function ArVar(msa::Array{Ti,2},W,lambdaH::T,lambdaJ::T,lambdaG::T; d = 2) where {T, Ti}
        q = maximum(msa)
        L,M = size(msa)
        δ = zeros(Ti, M, L, q)  # Ensure δ is a 3D array

        for b in 1:q, m in 1:M, k in 1:L  # Use Julia's efficient loop syntax
            δ[m, k, b] = (msa[k, m] == b)  # Parentheses for clarity
        end

        W = if sum(W) ≈ 1.0
            W
        else
            W./sum(W)
        end

        f1 = [sum(view(msa,1,:).==c)/M for c in 1:q] 
        Y = get_pca_components(msa,d=d)

        return new{T, Ti}(q,L,M,d,msa,W,Y,lambdaH::T,lambdaJ::T,lambdaG::T,δ,f1)  # Correctly specify type parameters
    end

    function ArVar(msa::Array{Ti,2},W,Y,lambdaH::T,lambdaJ::T,lambdaG::T; d = 2) where {T, Ti}
        q = maximum(msa)
        L,M = size(msa)
        δ = zeros(Ti, M, L, q)  # Ensure δ is a 3D array

        for b in 1:q, m in 1:M, k in 1:L  # Use Julia's efficient loop syntax
            δ[m, k, b] = (msa[k, m] == b)  # Parentheses for clarity
        end

        W = if sum(W) ≈ 1.0
            W
        else
            W./sum(W)
        end

        f1 = [sum(view(msa,1,:).==c)/M for c in 1:q] 

        return new{T, Ti}(q,L,M,d,msa,W,Y,lambdaH::T,lambdaJ::T,lambdaG::T,δ,f1)  # Correctly specify type parameters
    end
 
end


struct ArAlg
    method::Symbol
    verbose::Bool
    epsconv::Float64
    maxit::Int
end