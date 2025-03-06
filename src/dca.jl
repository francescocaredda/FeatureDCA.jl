function epistatic_score(net::pcaDCA.ArNet, var::pcaDCA.ArVar, idx::Int)
    arvar = ArDCA.ArVar(var.L, var.M, var.q, var.lambdaJ, var.lambdaH, var.msa, var.weights, 1.0, :NATURAL)
    H = net.H[2:end]
    J = net.J
    J_ = [permutedims(J[i], (2, 3, 1)) for i in 1:length(net.J)]
    arnet = ArDCA.ArNet([1:var.L;], net.f1, J_, H)
    return ArDCA.epistatic_score(arnet, arvar, idx)
end

function compute_PPV(score::Vector{Tuple{Int,Int,Float64}}, filestruct::String; min_separation::Int = 6)
    dist = compute_residue_pair_dist(filestruct)
    return map(x->x[4], compute_referencescore(score, dist, mindist = min_separation))
end

function compute_residue_pair_dist(filedist::String)
    d = readdlm(filedist)
    if size(d,2) == 4 
        return Dict((round(Int,d[i,1]),round(Int,d[i,2])) => d[i,4] for i in 1:size(d,1) if d[i,4] != 0)
    elseif size(d,2) == 3
        Dict((round(Int,d[i,1]),round(Int,d[i,2])) => d[i,3] for i in 1:size(d,1) if d[i,3] != 0)
    end

end

function compute_referencescore(score,dist::Dict; mindist::Int=6, cutoff::Number=8.0)
    nc2 = length(score)
    out = Tuple{Int,Int,Float64,Float64}[]
    ctrtot = 0
    ctr = 0
    for i in 1:nc2
        sitei,sitej,plmscore = score[i][1],score[i][2], score[i][3]
        dij = if haskey(dist,(sitei,sitej)) 
            dist[(sitei,sitej)]
        else
           continue
        end
        if sitej - sitei >= mindist 
            ctrtot += 1
            if dij < cutoff
                ctr += 1
            end
            push!(out,(sitei,sitej, plmscore, ctr/ctrtot))
        end
    end 
    out
end
