function pl_and_grad_first!(grad,x,arvar)
    T = eltype(x)
    @extract arvar: q L M d msa weights Y δ lambdaG f1
    G = reshape(x,q,d)

    mat_ene = zeros(q,M)

    for m = 1:M
        for a = 1:q
            mat_ene[a,m] = log.(f1[a]) + view(G,a,:)'Y[:,m] #can this be done with a multiplication?
        end
    end

    Z = sumexp(mat_ene, dims=1)
    logZ = log.(Z)
    pl = zero(T)
    @simd for m in eachindex(logZ)
        pl -= weights[m]*(mat_ene[msa[1,m],m] - logZ[m])
    end
    

    pl += lambdaG*(sum(abs2, G))

    grad .= zero(T)

    prob = exp.(mat_ene)./Z'

    ∂ = zeros(q,M)
    for a in 1:q
        for m in 1:M
            ∂[a,m] = weights[m]*((msa[1,m] == a) - prob[a,m])
        end
    end

    #gradient in G
    grad .-= reshape((∂*Y'),d*q)
    grad .+= 2*lambdaG*x
    

    return pl

end

function pl_and_grad!(grad,x,arvar)
    T = eltype(x)
    @extract arvar: q L M d msa weights Y δ lambdaH lambdaJ lambdaG
    N = length(x)
    it_h = 1:q
    it_G = q+1:d*q+q
    it_J = d*q+q+1:N
    s = Int.((N - d*q - q)/q^2)
    
    h = x[it_h]
    G = reshape(x[it_G],q,d)
    J = reshape(x[it_J],s,q,q)
    δ = view(δ, :, 1:s, :)
    

    mat_ene = zeros(q,M)
    # mat_ene = G'Y
    for m = 1:M
        for a = 1:q
            mat_ene[a,m] = h[a] + view(G,a,:)'view(Y,:,m) #can this be done with a multiplication?
            for k = 1:s
                mat_ene[a,m] += J[k,a,msa[k,m]]
            end
        end
    end

    Z = sumexp(mat_ene, dims=1)
    # return Z
    logZ = log.(Z)
    pl = zero(T)
    @simd for m in eachindex(logZ)
        pl -= weights[m]*(mat_ene[msa[s+1,m],m] - logZ[m])
    end
    
    pl += lambdaH*(sum(abs2, h))
    pl += lambdaJ*(sum(abs2, J))
    pl += lambdaG*(sum(abs2, G))

    grad .= zero(T)
    #softmax!(mat_ene, dims = 1) 
    prob = exp.(mat_ene)./Z'

    ∂ = zeros(q,M)
    for a in 1:q
        for m in 1:M
            ∂[a,m] = weights[m]*((msa[s+1,m] == a) - prob[a,m])
        end
    end
    #gradient in h

    for m in 1:M
        for a in 1:q
            grad[a] -= ∂[a,m]
        end
    end

    #gradient in G
    grad[it_G] .-= reshape((∂*Y'),d*q)

    #gradient in J
    g = zeros(T,s,q,q)
 

    @tullio g[k,a,b] := δ[m,k,b] * ∂[a,m] avx=true threads=true

    

    @views @inbounds grad[it_J] .-= vec(g)

    for i in it_h
        grad[i] += 2*lambdaH*x[i]
    end
    for i in it_G
        grad[i] += 2*lambdaG*x[i]
    end
    for i in it_J
        grad[i] += 2*lambdaJ*x[i]
    end

    # grad .*=-1.0

    return pl
end

function pl_and_grad_faster!(grad,x,arvar) #actual slower and more allocations
    T = eltype(x)
    @extract arvar: q L M d msa weights Y lambdaH lambdaJ lambdaG
    N = length(x)
    it_h = 1:q
    it_G = q+1:d*q+q
    it_J = d*q+q+1:N
    s = Int.((N - d*q - q)/q^2)
    
    h = x[it_h]
    G = reshape(x[it_G],q,d)
    J = reshape(x[it_J],s,q,q)
    
    pl = zero(T)

    pl += lambdaH*(sum(abs2, h))
    pl += lambdaJ*(sum(abs2, J))
    pl += lambdaG*(sum(abs2, G))

    grad .= zero(T)

    for i in it_h
        grad[i] += 2*lambdaH*x[i]
    end
    for i in it_G
        grad[i] += 2*lambdaG*x[i]
    end
    for i in it_J
        grad[i] += 2*lambdaJ*x[i]
    end

    vecene = zeros(Float64, q)
    Ym = zeros(Float64, d)
    partial_delta = zeros(Float64, q)

    @inbounds for m in 1:M
        Zm = view(msa, :, m)
        δ_ = view(arvar.δ, m, 1:s, :)
        Ym = Y[:,m]

        fillvecene!(vecene, h, G, J, Ym, Zm)

        Z = sumexp(vecene)
        pl -= weights[m]*(vecene[Zm[s+1]] - log(Z))

        softmax!(vecene)

        for a in 1:q
            partial_delta[a] = weights[m]*((Zm[s+1] == a) - vecene[a])
            grad[a] -= partial_delta[a]
        end

        @inbounds @views begin
            for k in 1:d
                for a in 1:q
                    grad[q + a + (k - 1) * q] -= partial_delta[a] * Ym[k]
                end
            end
        end
        @inbounds @views begin
            for b in 1:q   # Iterate over b
                for a in 1:q   # Iterate over a
                    for k in 1:s  # Iterate over k
                        index = q + d*q + k + (a - 1) * s + (b - 1) * s * q 
                        grad[index] -= δ_[k, b] * partial_delta[a]
                    end
                end
            end
        end
    end

    return pl
end

function fillvecene!(vecene, h, G, J, Ym, msa)
    vecene .= h + G*Ym 
    for a in axes(J,2)
        for k in axes(J,1)
            vecene[a] += J[k,a,msa[k]]
        end
    end
end


function optimfunwrapper(x::Vector, g::Vector, var)
    g === nothing && (g = zeros(Float64, length(x)))
    return pl_and_grad!(g, x, var)
end


function minimize_arnet(alg::ArAlg, var::ArVar{T,Ti}) where {T,Ti}
    @extract var : q L M d msa Y δ f1
    @extract alg : epsconv maxit method
    θ = Vector{Float64}(undef, q*L + L*d*q + div(L*(L-1),2)*q*q)
    vecps = Vector{Float64}(undef,L)
    ### OPTIMIZE FIRST SITE, ONLY G WITH H === F1
    x0_1 = zeros(Float64, d*q) #or zero or rand?????
    opt_1 = Opt(method, length(x0_1))
    ftol_abs!(opt_1, epsconv)
    xtol_rel!(opt_1, epsconv)
    xtol_abs!(opt_1, epsconv)
    ftol_rel!(opt_1, epsconv)
    maxeval!(opt_1, maxit)
    min_objective!(opt_1, (x, g) -> pl_and_grad_first!(g, x, var))
    elapstime_1 = @elapsed (minf_1, minx_1, ret_1) = optimize(opt_1, x0_1)
    alg.verbose && @printf("site = %d\tpl = %.4f\ttime = %.4f\t", 1, minf_1, elapstime_1)
    alg.verbose && println("status = $ret_1")
    θ[1:q] .= log.(f1)
    θ[q+1:q+d*q] .= minx_1
    vecps[1] = minf_1
    
    ### OPTIMIZE REMAINING SITES
    @threads for site in 2:L
        N = q + d*q + (site-1)*q*q
        x0 = zeros(Float64, N) #or rand?????
        opt = Opt(method, length(x0))
        ftol_abs!(opt, epsconv)
        xtol_rel!(opt, epsconv)
        xtol_abs!(opt, epsconv)
        ftol_rel!(opt, epsconv)
        maxeval!(opt, maxit)
        min_objective!(opt, (x, g) -> pl_and_grad!(g, x, var))
        elapstime = @elapsed (minf, minx, ret) = optimize(opt, x0)
        L1 = (site-1)*q+(site-1)*d*q+div((site-2)*(site-1),2)*q*q+1
        L2 = site*q+site*d*q+div(site*(site-1),2)*q*q
        alg.verbose && @printf("site = %d\tpl = %.4f\ttime = %.4f\t", site, minf, elapstime)
        alg.verbose && println("status = $ret")
        θ[L1:L2] .= minx
        vecps[site] = minf
    end
 
    H, G, J = unpack_params(θ,var)
    
    return ArNet(H,G,J,f1),vecps
end


function trainer(fasta::String;
    Y::Symbol=:PCA,
    lambdaH::Float64=0.01,
    lambdaG::Float64=0.01,
    lambdaJ::Float64=0.01,
    epsconv::Float64=1e-5,
    maxit::Int=1000,
    verbose::Bool=true,
    method=:LD_LBFGS)

    alg = ArAlg(method, verbose, epsconv, maxit)
    W,Z,_,_,_ = read_fasta(fasta)

    arvar = if Y == :PCA
        ArVar(Z,W,lambdaH,lambdaJ,lambdaG)
    elseif Y == :ZERO
        ArVar(Z,W,zeros(2,length(W)),lambdaH,lambdaJ,lambdaG)
    else 
        error("Wrong value for Y, it can be either :PCA or :ZERO")
    end

    ArVar(Z,W,lambdaH,lambdaJ,lambdaG)

    arnet,vecps = minimize_arnet(alg, arvar)
    
    return arnet, arvar, vecps

end

function trainer(Z::Matrix,W::Array{Float64,1};
    Y::Symbol=:PCA,
    lambdaH::Float64=0.01,
    lambdaG::Float64=0.01,
    lambdaJ::Float64=0.01,
    epsconv::Float64=1e-5,
    maxit::Int=1000,
    verbose::Bool=true,
    method=:LD_LBFGS)

    alg = ArAlg(method, verbose, epsconv, maxit)

    arvar = if Y == :PCA
        ArVar(Z,W,lambdaH,lambdaJ,lambdaG)
    elseif Y == :ZERO
        ArVar(Z,W,zeros(2,length(W)),lambdaH,lambdaJ,lambdaG)
    else 
        error("Wrong value for Y, it can be either :PCA or :ZERO")
    end

    ArVar(Z,W,lambdaH,lambdaJ,lambdaG)

    arnet,_ = minimize_arnet(alg, arvar)
    
    return arnet, arvar

end

#script to sample and check, to be implemented in a notebook for future references
# Zs = sample(net,Y_)
# Zs_pc = predict(M, pcaDCA.one_hot(Zs)) 
# close("all") 
# plot_density(Zs_pc[1,:],Zs_pc[2,:])
# gcf()