function likelihood(net, seq::Vector, y::Vector)
    @extract net:H G J
    L = length(H)
    q = length(H[1])
    pl = zeros(L)
    e = H[1] .+ G[1]*y #q dimensional vector
    softmax!(e)
    pl[1] = log(e[seq[1]])
    @threads for site in 2:L
        e_ = zeros(q)
        Js = J[site-1]
        Gs = G[site]
        e_ .= H[site] .+ Gs*y
        for j in 1:site-1
            for a in 1:q
                e_[a] += Js[j,a, seq[j]]
            end
        end
        softmax!(e_)
        pl[site] = log(e_[seq[site]])
    end
    return sum(pl)
end

function likelihood(net, msa::Matrix, Y::Matrix)
    @extract net:H G J
    L, M = size(msa)
    q = length(H[1])
    pl = zeros(M)

    @threads for m in 1:M
        pl_scra = 0.0
        y = Y[:, m]
        seq = msa[:, m]

        # Thread-local buffer for `e` to avoid race conditions
        e = H[1] .+ G[1] * y
        
        softmax!(e)
        pl_scra += log(e[seq[1]])

        for site in 2:L
            Js = J[site-1]
            Gs = G[site]
            
            # Update `e` safely within the thread
            e .= H[site] .+ Gs * y
            
            for j in 1:site-1
                for a in 1:q
                    e[a] += Js[j, a, seq[j]]
                end
            end

            softmax!(e)
            pl_scra += log(e[seq[site]])
        end

        # Safe write to `pl[m]`
        pl[m] = pl_scra
    end

    return sum(pl) / M
end

function likelihood(net, msa::Matrix, W::Vector, Y::Matrix)
    @extract net:H G J
    L, M = size(msa)
    q = length(H[1])
    pl = zeros(M)

    @threads for m in 1:M
        pl_scra = 0.0
        y = Y[:, m]
        seq = msa[:, m]

        # Thread-local buffer for `e` to avoid race conditions
        e = H[1] .+ G[1] * y
        
        softmax!(e)
        pl_scra += log(e[seq[1]])

        for site in 2:L
            Js = J[site-1]
            Gs = G[site]
            
            # Update `e` safely within the thread
            e .= H[site] .+ Gs * y
            
            for j in 1:site-1
                for a in 1:q
                    e[a] += Js[j, a, seq[j]]
                end
            end

            softmax!(e)
            pl_scra += log(e[seq[site]])
        end

        # Safe write to `pl[m]`
        pl[m] = W[m]*pl_scra
    end

    return sum(pl)
end

likelihood(net, msa::Matrix, Y::Vector) = likelihood(net, msa, hcat(fill(Y, size(msa,2))...))


function energy(net, seq::Vector, y::Vector)
    @extract net:H G J
    L = length(H)
    ene = zeros(L)
    ene[1] = -H[1][seq[1]] - G[1][seq[1],:]'y #q dimensional vector
    @threads for site in 2:L

        Js = J[site-1]
        Gs = G[site]
        ene[site] -= H[site][seq[site]] + Gs[seq[site],:]'y
        for j in 1:site-1
            ene[site] -= Js[j,seq[site], seq[j]]
        end
    end
    return sum(ene)
end

function energy(net, msa::Matrix, Y::Matrix)
    @extract net:H G J
    L, M = size(msa)
    q = length(H[1])
    ene = zeros(M)

    @threads for m in 1:M
        y = Y[:, m]
        seq = msa[:, m]
        ene_scra = 0.0  # Thread-local accumulation variable
        
        ene_scra -= H[1][seq[1]] + G[1][seq[1], :]' * y  # q-dimensional vector
        
        for site in 2:L
            Js = J[site-1]
            Gs = G[site]
            ene_scra -= H[site][seq[site]] + Gs[seq[site], :]' * y
            for j in 1:site-1
                ene_scra -= Js[j, seq[site], seq[j]]
            end
        end
        
        ene[m] = ene_scra  # Safe write to unique index
    end

    return sum(ene) / M
end


energy(net, msa::Matrix, Y::Vector) = energy(net, msa, hcat(fill(Y, size(msa,2))...))

function likelihood_noY(net, seq::Vector)
    @extract net:H J
    L = length(H)
    q = length(H[1])
    pl = zeros(L)
    e = H[1] #q dimensional vector
    softmax!(e)
    pl[1] = log(e[seq[1]])
    @threads for site in 2:L
        e_ = zeros(q)
        Js = J[site-1]
        e_ .= H[site]
        for j in 1:site-1
            for a in 1:q
                e_[a] += Js[j,a, seq[j]]
            end
        end
        softmax!(e_)
        pl[site] = log(e_[seq[site]])
    end
    return sum(pl)
end

function likelihood_noY(net, msa::Matrix)
    @extract net:H J
    L, M = size(msa)
    q = length(H[1])
    pl = zeros(M)
    
    @threads for m in 1:M
        pl_scra = 0.0
        seq = msa[:, m]
        
        # Allocate thread-local buffer to prevent race conditions
        e = copy(H[1])
        
        softmax!(e)
        pl_scra += log(e[seq[1]])
        
        for site in 2:L
            fill!(e, 0.0)  # Zero out instead of re-allocating
            Js = J[site-1]
            e .= H[site]
            
            for j in 1:site-1
                for a in 1:q
                    e[a] += Js[j, a, seq[j]]
                end
            end
            
            softmax!(e)
            pl_scra += log(e[seq[site]])
        end
        
        pl[m] = pl_scra  # Safe write to unique index
    end
    
    return sum(pl) / M
end


function energy_noY(net, seq::Vector)
    @extract net:H J
    L = length(H)
    ene = zeros(L)
    ene[1] = -H[1][seq[1]] #q dimensional vector
    @threads for site in 2:L
        Js = J[site-1]
        ene[site] -= H[site][seq[site]]
        for j in 1:site-1
            ene[site] -= Js[j,seq[site], seq[j]]
        end
    end
    return sum(ene)
end

function energy_noY(net, msa::Matrix)
    @extract net:H J
    L, M = size(msa)
    q = length(H[1])
    ene = zeros(M)

    @threads for m in 1:M
        seq = msa[:, m]
        ene_scra = 0.0  # Thread-local accumulation variable
        
        ene_scra -= H[1][seq[1]]
        
        for site in 2:L
            Js = J[site-1]
            ene_scra -= H[site][seq[site]]
            
            for j in 1:site-1
                ene_scra -= Js[j, seq[site], seq[j]]
            end
        end
        
        ene[m] = ene_scra  # Safe write to unique index
    end

    return sum(ene) / M
end

entropy(net, msa, Y::Matrix) = -likelihood(net, msa, Y)/ size(msa, 1)
entropy(net, msa, Y::Vector) = -likelihood(net, msa, Y)/ size(msa, 1)

entropy_noY(net, msa) = -likelihood_noY(net, msa)/ size(msa, 1)