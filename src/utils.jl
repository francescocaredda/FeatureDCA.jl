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

function read_annotated_fasta(filename::AbstractString; theta::Any=:auto, remove_dups::Bool=true, max_gap_fraction::Real=1.0)
    Z = read_fasta_alignment(filename, max_gap_fraction)
    seqs = FastaIO.readfasta(filename)
    annotations = [seqs[i][1:end-1] for i in axes(Z,2)]
    if remove_dups
        Z, z = remove_duplicate_sequences(Z)
        N, M = size(Z)
        q = round(Int, maximum(Z))
        W, Meff = compute_weights(Z, theta)
        return W, Z, N, M, q, annotations[z]
    else
        N, M = size(Z)
        q = round(Int, maximum(Z))
        W, Meff = compute_weights(Z, theta)
        return W, Z, N, M, q, annotations
    end

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


function regularization(net::FeatureDCA.ArNet, var)
    @extract net: H G J f1
    @extract var: lambdaH lambdaG lambdaJ
    return lambdaH*sum(abs2,vcat(H[2:end]...)) + lambdaG*sum(abs2,vcat(G[:]...)[:]) + lambdaJ*sum(abs2,vcat(J[:]...)[:])
end


const AA_MAP = Dict(
                1 => 'A',  2 => 'C',  3 => 'D',  4 => 'E',  5 => 'F',
                6 => 'G',  7 => 'H',  8 => 'I',  9 => 'K', 10 => 'L',
                11 => 'M', 12 => 'N', 13 => 'P', 14 => 'Q', 15 => 'R',
                16 => 'S', 17 => 'T', 18 => 'V', 19 => 'W', 20 => 'Y',
                21 => '-')

function convert_to_amino_acids(sequence::Vector{Int})
    return join([AA_MAP[n] for n in sequence])
end
                
function convert_to_amino_acids_ignore_gaps(sequence::Vector{Int})
    return join([AA_MAP[n] for n in sequence if n != 21])
end

function create_fasta(Z, filename; gaps = false, annotations=nothing)
    
    f = gaps ? convert_to_amino_acids : convert_to_amino_acids_ignore_gaps

    annotations = annotations === nothing ? ["seq$(i)" for i in axes(Z,2)] : [annotations[i][1] for i in axes(Z,2)]
    sequences = [(annotations[i], f(Z[:,i])) for i in axes(Z,2)]

    writefasta(filename, sequences)
    println("Fasta file $filename created with $(length(sequences)) sequences.")

end

function get_fasta_ids(file_path::String)
    ids = String[]
    open(file_path, "r") do file
        for line in eachline(file)
            if startswith(line, ">")
                # Remove the ">" character and any surrounding whitespace.
                push!(ids, strip(line[2:end]))
            end
        end
    end
    return ids
end

function read_rmsd_matrix(file_path::String)
    # Read the file into an array of lines
    lines = readlines(file_path)
    
    # Ensure the file is not empty
    if isempty(lines)
        error("The input file is empty.")
    end
    
    # Parse the header row (skip the first element, which is an empty label)
    header = split(lines[1], ",")[2:end]
    
    # Initialize an empty matrix to store RMSD values
    num_queries = length(lines) - 1
    num_targets = length(header)
    
    # Preallocate the matrix
    rmsd_matrix = zeros(num_queries, num_targets)
    
    # Read each subsequent line
    for (i, line) in enumerate(lines[2:end])
        values = split(line, ",")
        rmsd_matrix[i, :] = parse.(Float64, values[2:end])  # Convert values to Float64
    end
    
    return rmsd_matrix
end

function pca2_wasserstein(Y1::AbstractMatrix, Y2::AbstractMatrix; e=1e-1, maxiter=2000, atol=1e-9, rtol=1e-9)
           
    M1 = size(Y1, 2)
    M2 = size(Y2, 2)

    μ = fill(1/M1, M1)
    ν = fill(1/M2, M2)

    Cxy = pairwise(Euclidean(), Y1, Y2)  # M1×M2
    Cxx = pairwise(Euclidean(), Y1, Y1)  # M1×M1
    Cyy = pairwise(Euclidean(), Y2, Y2)  # M2×M2

    return sinkhorn_divergence(μ, ν, Cxy, Cxx, Cyy, e; maxiter=maxiter, atol=atol, rtol=rtol)
end

"""
    dms_single_site(arnet::ArNet, arvar::ArVar, seqid::Int; pc::Float64=0.1)
    
Return a `q×L` matrix of containing `-log(P(mut))/log(P(seq))` for all single
site mutants of the reference sequence `seqid`, and a vector of the indices of
the residues of the reference sequence that contain gaps (i.e. the 21
amino-acid) for which the score has no sense and is set by convention to `+Inf`.
A negative value indicate a beneficial mutation, a value 0 indicate
the wild-type amino-acid.
"""
function dms_single_site(arnet::ArNet, arvar::ArVar, seqid::Int)
    @extract arnet:H J G
    @extract arvar:msa L M q d Y

    pca_map = fit(PCA, one_hot(msa), maxoutdim=d)

    1 ≤ seqid ≤ M || error("seqid=$seqid should be in the interval [1,...,$M]")

    Da = fill(Inf64, q, L)
    xori = msa[:, seqid]
    yori = Y[:, seqid]
    xmut = copy(xori)
    ymut = copy(yori)
    idxnogap = findall(x -> x != 21, xori)
    
    ll0 = -likelihood(arnet, xori, yori)

    @inbounds for i in idxnogap
        if xori[i] == 21
            continue
        end
        for a in 1:q
            if a != xori[i]
                xmut[i] = a
                ymut = predict(pca_map, one_hot(xmut)')[:]
                Da[a, i] = -likelihood(arnet, xmut, ymut) - ll0
            else
                Da[a, i] = 0.0
            end
        end
        xmut[i] = xori[i] #reset xmut to the original velue 
    end
    return Da, sort!(setdiff(1:L, idxnogap))
end

function pca_dms(Z, wt_idx; pca_components = [1,2])
    L,_ = size(Z)
    M = fit(PCA, one_hot(Z), maxoutdim=21*L, pratio=1.0)
    y_wt = predict(M, one_hot(Z[:, wt_idx:wt_idx]))


    wt = Z[:, wt_idx]
    idxnogap = findall(x -> x != 21, wt)

    DMS = zeros(Int, L, length(idxnogap)*20)
    DMS[:,1:end] .= wt
    for i in axes(idxnogap,1)
        for a in 1:20
            DMS[idxnogap[i],(i-1)*20 + a] = a
        end
    end

    y_dms = predict(M, one_hot(DMS))

    d = pairwise(Euclidean(), y_wt[pca_components,:], y_dms[pca_components,:])[:]
    

    return d
end

function pca_dms(Z, M, wt_idx; pca_components = [1,2])
    L,_ = size(Z)
    y_wt = predict(M, one_hot(Z[:, wt_idx:wt_idx]))


    wt = Z[:, wt_idx]
    idxnogap = findall(x -> x != 21, wt)

    DMS = zeros(Int, L, length(idxnogap)*20)
    DMS[:,1:end] .= wt
    for i in axes(idxnogap,1)
        for a in 1:20
            DMS[idxnogap[i],(i-1)*20 + a] = a
        end
    end

    y_dms = predict(M, one_hot(DMS))

    d = pairwise(Euclidean(), y_wt[pca_components,:], y_dms[pca_components,:])[:]
    

    return d
end