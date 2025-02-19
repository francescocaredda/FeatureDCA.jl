function logsumexp(x::AbstractArray; dims=1)
    m = maximum(x, dims=dims)
    return dropdims(m .+ log.(sum(exp.(x .- m), dims=dims)), dims=dims)
end

function logsumexp(x::AbstractVector; dims=1)
    m = maximum(x)
    return m .+ log.(sum(exp.(x .- m)))
end

function sumexp(a::AbstractArray{<:Real}; dims=1)
    m = maximum(a; dims=dims)
    return dropdims(exp.(m) .* sum(exp.(a .- m); dims=dims), dims=dims)
end

function sumexp(a::AbstractVector{<:Real})
    m = maximum(a)
    return exp(m) * sum(exp.(a .- m))
end

function likelihood(x,arvar) #for testing automatic differentiation #for a single site
    @extract arvar: q L M d msa weights Y δ lambdaH lambdaJ lambdaG
    N = length(x)
    it_h = 1:q
    it_G = q+1:d*q+q
    it_J = d*q+q+1:N
    s = Int.((N - d*q - q)/q^2)
    h = x[it_h]
    G = reshape(x[it_G],q,d)
    J = reshape(x[it_J],s,q,q)


    mat_ene = [h[a] + view(G,a,:)'Y[:,m] + sum([J[k,a,msa[k,m]] for k = 1:s]) for a = 1:q, m = 1:M]
    Z = logsumexp(mat_ene, dims=1)
    pl = [weights[m]*(mat_ene[msa[s+1,m],m] - Z[m]) for m in 1:M]
    
    return -sum(pl) + lambdaH*sum(abs2,h) + lambdaG*sum(abs2,G) + lambdaJ*sum(abs2,J)
    
end

function likelihood_first(x,arvar) #for testing automatic differentiation
    @extract arvar: q L M d msa weights Y lambdaG f1
    
    G = reshape(x,q,d)
    
    mat_ene = [log(f1[a]) + view(G,a,:)'Y[:,m] for a = 1:q, m = 1:M]
    Z = logsumexp(mat_ene, dims=1)
    pl = [weights[m]*(mat_ene[msa[1,m],m] - Z[m]) for m in 1:M]
    
    return -sum(pl) + lambdaG*sum(abs2,G)
    
end

function regularization(x,arvar) #for testing automatic differentiation
    @extract arvar: q L M d msa Y δ lambdaH lambdaJ lambdaG
    N = length(x)
    it_h = 1:q
    it_G = q+1:d*q+q
    it_J = d*q+q+1:N
    
    return lambdaH*sum(abs2,x[it_h]) + lambdaG*sum(abs2,x[it_G]) + lambdaJ*sum(abs2,x[it_J])
end

function read_fasta(filename::AbstractString; max_gap_fraction::Real=0.9, theta::Any=:auto, remove_dups::Bool=true)
    Z = read_fasta_alignment(filename, max_gap_fraction)
    if remove_dups
        Z, _ = remove_duplicate_sequences(Z)
    end
    N, M = size(Z)
    q = round(Int, maximum(Z))
    W, Meff = compute_weights(Z, theta)
    return W, Z, N, M, q
end

function read_annotated_fasta(filename::AbstractString; theta::Any=:auto, remove_dups::Bool=true)
    Z = read_fasta_alignment(filename, 1.0)
    seqs = FastaIO.readfasta(filename)
    annotations = [seqs[i][1:end-1] for i in axes(Z,2)]
    if remove_dups
        Z, z = remove_duplicate_sequences(Z)
    end
    N, M = size(Z)
    q = round(Int, maximum(Z))
    W, Meff = compute_weights(Z, theta)
    return W, Z, N, M, q, annotations[z]
   

end

function one_hot(msa::Array{T,2}; q = 21) where T
    N, M = size(msa)
    new_msa = zeros(N*q,M)
    for i in 1:N
        for j in 1:M
            index = msa[i,j]  
            new_msa[(i-1)*q + index, j] = 1
        end
    end
    return new_msa
end

function one_hot(seq::Array{T,1}; q = 21) where T
    N = length(seq)
    new_msa = zeros(N*q)
    for j in 1:N
        index = seq[j]  
        new_msa[(j-1)*q + index] = 1
    end
    return Matrix{}(new_msa')
end

function get_pca_components(Z; d=2)
    q = maximum(Z)
    Z_one_hot = one_hot(Z,q=q)
    M = fit(PCA, Z_one_hot, maxoutdim=d)
    Z_pc = predict(M, Z_one_hot)
    return Z_pc

end 


function unpack_params(θ, arvar::ArVar)
    @extract arvar: q L d 

    arrJ = Array{Float64,3}[]
    arrH = Vector{Float64}[]
    arrG = Array{Float64,2}[]
    # ctr = 0
    push!(arrH, θ[1:q])
    push!(arrG, reshape(θ[q+1:q+d*q],q,d))
    ctr = q+d*q
    for site in 2:L
        _arrH = zeros(q)
        for a in 1:q
            ctr += 1
            _arrH[a] = θ[ctr]
        end
        push!(arrH, _arrH)
        
        _arrG = zeros(q, d)
        for k in 1:d
            for a in 1:q
                ctr += 1
                _arrG[a, k] = θ[ctr]
            end
        end
        push!(arrG, _arrG)
        
        
        _arrJ = zeros(length(1:site-1), q, q)
        for b in 1:q
            for a in 1:q
                for i in 1:site-1
                    ctr += 1
                    _arrJ[i, a, b] = θ[ctr]
                end
            end
        end
        push!(arrJ, _arrJ)
    end
    @assert ctr == length(θ)
    return arrH, arrG, arrJ
end

log0(x::Number) = x > 0 ? log(x) : zero(x)

softmax(x::AbstractArray{T}; dims = 1) where {T} = softmax!(similar(x, float(T)), x; dims)

softmax!(x::AbstractArray; dims = 1) = softmax!(x, x; dims)

function softmax!(out::AbstractArray{T}, x::AbstractArray; dims = 1) where {T}
    max_ = maximum(x; dims)
    if all(isfinite, max_)
        @fastmath out .= exp.(x .- max_)
    else
        @fastmath @. out = ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 1, 0), exp(x - max_))
    end
    out ./= sum(out; dims)
end


function sample(arnet::ArNet, Y::Array{Float64,2})
    @extract arnet:H G J f1
    msamples = size(Y, 2)
    q = length(H[1])
    N = length(H) 
    res = Matrix{Int}(undef, N , msamples)
    e1 = H[1] .+ G[1]*Y
    softmax!(e1)
    Threads.@threads for m in 1:msamples

        totH = Vector{Float64}(undef, q)
        sample_z = Vector{Int}(undef, N)
        sample_z[1] = wsample(1:q, view(e1, :, m))

        for site in 2:N
            Js = J[site-1]
            Gs = G[site]
            totH = H[site] .+ Gs*Y[:,m]
            @turbo for j in 1:site-1
                for a in 1:q
                    totH[a] += Js[j,a, sample_z[j]]
                end
            end
            p = softmax(totH)
            sample_z[site] = wsample(1:q, p)
        end
        res[:, m] .= sample_z
    end
    return res
end
function sample(arnet::ArNet, Y::Vector{Float64}, msamples)
    @extract arnet:H G J f1
    q = length(H[1])
    N = length(H) 
    res = Matrix{Int}(undef, N , msamples)
    e1 = H[1] .+ G[1]*Y
    softmax!(e1)
    Threads.@threads for m in 1:msamples

        totH = Vector{Float64}(undef, q)
        sample_z = Vector{Int}(undef, N)
        sample_z[1] = wsample(1:q, view(e1, :))

        for site in 2:N
            Js = J[site-1]
            Gs = G[site]
            totH = H[site] .+ Gs*Y
            @turbo for j in 1:site-1
                for a in 1:q
                    totH[a] += Js[j,a, sample_z[j]]
                end
            end
            p = softmax(totH)
            sample_z[site] = wsample(1:q, p)
        end
        res[:, m] .= sample_z
    end
    return res
end



compute_freq(Z::Matrix,W::Vector{Float64}) = compute_weighted_frequencies(Matrix{Int8}(Z), W)


"""
    compute_freq(Z::Matrix)
Function that returns a tuple with the single and pairwise frequencies of the alignment Z. The weight vector is set to 1/M, where M is the number of sequences in the alignment.
"""
compute_freq(Z::Matrix) = compute_weighted_frequencies(Matrix{Int8}(Z), fill(1/size(Z,2), size(Z,2)))

function energy(seq::AbstractVector,y::AbstractVector,arnet::pcaDCA.ArNet)
    @extract arnet:H G J
    L = length(H)
    q = length(H[1])
    e = H[1] .+ G[1]*y #q dimensional vector
    softmax!(e)
    pl = -log(e[seq[1]])
    for site in 2:L
        Js = J[site-1]
        Gs = G[site]
        e .= H[site] .+ Gs*y
        for j in 1:site-1
            for a in 1:q
                e[a] += Js[j,a, seq[j]]
            end
        end
        softmax!(e)
        pl -= log(e[seq[site]])
    end
    return pl
end

function energy(msa::AbstractArray, Y::AbstractMatrix, weigths::AbstractVector, net::pcaDCA.ArNet)
    plvec = zeros(size(msa, 2))
    @threads for m in axes(msa, 2)
        plvec[m] = energy(msa[:, m], Y[:, m], net)
    end
    return plvec.*weigths
end

energy(msa, Y, net::pcaDCA.ArNet, var::pcaDCA.ArVar) = energy(msa, Y, var.weights, net)
energy(msa, Y, net::pcaDCA.ArNet) = energy(msa, Y, fill(1/size(msa, 2), size(msa,2)), net)

statistical_entropy(msa::AbstractArray, Y, net::pcaDCA.ArNet) = sum(energy(msa, Y, net))/size(msa,1)


function encode_amino_acids(seq::Vector{Int})
    # Define mapping from numbers (1-21) to amino acid letters
    num_to_aa = Dict(
        1 => 'A',  2 => 'C',  3 => 'D',  4 => 'E',  5 => 'F',
        6 => 'G',  7 => 'H',  8 => 'I',  9 => 'K', 10 => 'L',
       11 => 'M', 12 => 'N', 13 => 'P', 14 => 'Q', 15 => 'R',
       16 => 'S', 17 => 'T', 18 => 'V', 19 => 'W', 20 => 'Y',
       21 => '-'  # Assuming 21 represents a gap
    )

    # Convert sequence using the mapping
    return join(num_to_aa[n] for n in seq)
end


function site_likelihood(site, net::pcaDCA.ArNet, var::pcaDCA.ArVar; reg = true)
    
    @extract var: q L M d msa weights Y lambdaH lambdaG lambdaJ f1
    @extract net: H G J
    if site == 1
        G1 = G[1] 
        
        mat_ene = [log(f1[a]) + view(G1,a,:)'Y[:,m] for a = 1:q, m = 1:M]
        Z = logsumexp(mat_ene, dims=1)
        pl = [weights[m]*(mat_ene[msa[1,m],m] - Z[m]) for m in 1:M]
        
        res = if reg
            -sum(pl) + lambdaG*sum(abs2,G1)
        else
            -sum(pl)
        end
        return res
    else 
        h = net.H[site]
        G = net.G[site]
        J = net.J[site-1]
        mat_ene = [h[a] + view(G,a,:)'Y[:,m] + sum([J[k,a,msa[k,m]] for k = 1:site-1]) for a = 1:q, m = 1:M]
        Z = logsumexp(mat_ene, dims=1)
        pl = [weights[m]*(mat_ene[msa[site,m],m] - Z[m]) for m in 1:M]
    
        res = if reg
            -sum(pl) + lambdaH*sum(abs2,h) + lambdaG*sum(abs2,G) + lambdaJ*sum(abs2,J)
        else
            -sum(pl)
        end
        return res
    
    end
end

function likelihood(net::pcaDCA.ArNet, var::pcaDCA.ArVar; reg = true) 
    @extract var: L
    vecps = Vector{Float64}(undef,L)
    @threads for site in 1:L
        vecps[site] = site_likelihood(site, net, var, reg = reg) 
    end
    return sum(vecps)
end



function site_likelihood(site, net::pcaDCA.ArNet, msa, Y, weights)
    @extract net: H G J f1
    L, M = size(msa)
    q = length(H[1])

    if site == 1
        G1 = G[1]
        mat_ene = Matrix{Float64}(undef, q, M)

        @inbounds for a in 1:q
            @views G1a = G1[a, :]
            for m in 1:M
                mat_ene[a, m] = log(f1[a]) + dot(G1a, Y[:, m])
            end
        end

        Z = logsumexp(mat_ene, dims=1)
        pl = zero(Float64)

        @inbounds for m in 1:M
            pl += weights[m] * (mat_ene[msa[1, m], m] - Z[m])
        end

        return -pl
    else
        h = H[site]
        Gs = G[site]
        Js = J[site-1]
        mat_ene = Matrix{Float64}(undef, q, M)

        @inbounds for a in 1:q
            @views Gsa = Gs[a, :]
            for m in 1:M
                sum_J = zero(Float64)
                for k in 1:(site-1)
                    sum_J += Js[k, a, msa[k, m]]
                end
                mat_ene[a, m] = h[a] + dot(Gsa, Y[:, m]) + sum_J
            end
        end

        Z = logsumexp(mat_ene, dims=1)
        pl = zero(Float64)

        @inbounds for m in 1:M
            pl += weights[m] * (mat_ene[msa[site, m], m] - Z[m])
        end

        return -pl
    end
end

function likelihood(net::pcaDCA.ArNet, msa, Y, weights)
    L = length(net.H)
    vecps = Vector{Float64}(undef, L)

    @threads for site in 1:L
        vecps[site] = site_likelihood(site, net, msa, Y, weights)
    end

    return sum(vecps)
end

likelihood(net::pcaDCA.ArNet, msa, Y) = likelihood(net::pcaDCA.ArNet, msa, Y, fill(1/size(msa, 2), size(msa, 2)))

