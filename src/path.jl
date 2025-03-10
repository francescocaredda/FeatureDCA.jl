using PyPlot
import DensityPlot: plot_density

function simple_path_search(net, var; msample::Int = 1000, d::Int= 2, seq_start=rand(axes(var.msa,2)), seq_arrival=rand(axes(var.msa,2)))

    M = fit(PCA, pcaDCA.one_hot(var.msa), maxoutdim=d)
    ZPC = predict(M, pcaDCA.one_hot(var.msa))

    close("all")
    fig, ax1 = subplots(2, 1, figsize=(6, 8), gridspec_kw=Dict(:height_ratios => [2, 1]))
    plot_density(ZPC[1,:], ZPC[2,:], ax1[1])
    Zss = []
    y_steps = []
    likelihood_path = []
    η = 0.1
    y_i = ZPC[:,seq_start][:]
    y_f = ZPC[:,seq_arrival][:]
    Δ = y_f - y_i
    nΔ = norm(Δ)
    Δ ./= nΔ # unit vector in the direction of the target
    i = 0
    #println("iteration: ", i, " norm: ", nΔ)
    ax1[1].plot(y_i[1],y_i[2],".", c="b")
    # ax1[1].plot(y_f[1],y_f[2],".", c="r")
    #ax[1].arrow(y_i[1],y_i[2], y_f[1]-y_i[1], y_f[2]-y_i[2], head_width=0.2, head_length=0.3, fc="blue", ec="black")
    last_five = rand(5)
    println("Starting at ", ZPC[1:d,seq_start], " and going to ", ZPC[1:d,seq_arrival])
    while nΔ > 0.2 && i < 250
        i += 1
        y_step = y_i + η*Δ
        push!(y_steps, y_step)
        Zs_ = sample(net, y_step, msample)
        push!(Zss, Zs_)
        push!(likelihood_path, pcaDCA.likelihood(net, Zs_, y_step))
        # push!(likelihood_path, pcaDCA.likelihood_noY(net, Zs_))
        ZsPC_ = predict(M, pcaDCA.one_hot(Zs_))
        y_i = mean(ZsPC_, dims=2)[:]
        Δ = y_f - y_i
        nΔ = norm(Δ)
        Δ ./= nΔ

        popfirst!(last_five)
        push!(last_five, nΔ)
        η = if std(last_five) < 0.1
            η*(1.0 + std(last_five))
        else
            η
        end


        # println("iteration: ", i, " norm: ", nΔ, " std: ", std(last_five))
        println("step ", i, " position: ", y_i[1:d].-y_f[1:d])

        ax1[1].plot(y_i[1], y_i[2], ".", c="k")
        # ax1[1].plot(y_step[1], y_step[2], ".", c="g")
    end
    ax1[1].plot(y_f[1],y_f[2],"x", c="r")
    ax1[2].set_title("likelihood, sum = $(round(sum(likelihood_path), digits=2))")
    ax1[2].plot(likelihood_path, label="likelihood")
    # savefig("pca_path.pdf")
    # plt.tight_layout()
    # plt.show()
    subplots_adjust(hspace=0.35)  

    return Zss, y_steps, fig

end

function simple_path_search_no_plot(net, var; msample::Int = 1000, d::Int= 2, seq_start=rand(axes(var.msa,2)), seq_arrival=rand(axes(var.msa,2)))

    M = fit(PCA, pcaDCA.one_hot(var.msa), maxoutdim=d)
    ZPC = predict(M, pcaDCA.one_hot(var.msa))

    Zss = []
    y_steps = []
    likelihood_path = []
    η = 0.1
    y_i = ZPC[:,seq_start][:]
    y_f = ZPC[:,seq_arrival][:]
    Δ = y_f - y_i
    nΔ = norm(Δ)
    Δ ./= nΔ # unit vector in the direction of the target
    i = 0
    #println("iteration: ", i, " norm: ", nΔ)
    
    last_five = rand(5)
    println("Starting at ", ZPC[1:d,seq_start], " and going to ", ZPC[1:d,seq_arrival])
    while nΔ > 0.2 && i < 250
        i += 1
        y_step = y_i + η*Δ
        push!(y_steps, y_step)
        Zs_ = sample(net, y_step, msample)
        push!(Zss, Zs_)
        push!(likelihood_path, pcaDCA.likelihood(net, Zs_, y_step))
        # push!(likelihood_path, pcaDCA.likelihood_noY(net, Zs_))
        ZsPC_ = predict(M, pcaDCA.one_hot(Zs_))
        y_i = mean(ZsPC_, dims=2)[:]
        Δ = y_f - y_i
        nΔ = norm(Δ)
        Δ ./= nΔ

        popfirst!(last_five)
        push!(last_five, nΔ)
        η = if std(last_five) < 0.1
            η*(1.0 + std(last_five))
        else
            η
        end


        # println("iteration: ", i, " norm: ", nΔ, " std: ", std(last_five))
        println("step ", i, " position: ", y_i[1:d].-y_f[1:d])

        # ax1[1].plot(y_step[1], y_step[2], ".", c="g")
    end
     

    return Zss, y_steps, likelihood_path

end

function multiple_path(net, var; iterations = 20, msample::Int = 1000, d::Int= 2, seq_start=rand(axes(var.msa,2)), seq_arrival=rand(axes(var.msa,2)))

    M = fit(PCA, pcaDCA.one_hot(var.msa), maxoutdim=d)
    ZPC = predict(M, pcaDCA.one_hot(var.msa))

    close("all")
    fig, ax1 = subplots(2, 1, figsize=(6, 8), gridspec_kw=Dict(:height_ratios => [2, 1]))
    plot_density(ZPC[1,:], ZPC[2,:], ax1[1])

    likelihoods = zeros(iterations)
    
    i = 0
    # last_five = [rand(5) for _ in 1:iterations]
    y_ii = ZPC[:,seq_start][:]
    y_ff = ZPC[:,seq_arrival][:]
    ax1[1].plot(y_ii[1],y_ii[2],"x", c="r")
    ax1[1].plot(y_ff[1],y_ff[2],".", c="b")

    for m in 1:iterations 
        y_i = ZPC[:,seq_start][:]
        y_f = ZPC[:,seq_arrival][:]
        Δ = y_f - y_i
        nΔ = norm(Δ)
        Δ ./= nΔ # unit vector in the direction of the target
        i = 0
        η = 0.1
        last_five = rand(5)
        while nΔ > 0.2 && i < 250
            i += 1
            y_step = y_i + η*Δ
            Zs_ = sample(net, y_step, msample)
            likelihoods[m] += pcaDCA.likelihood(net, Zs_, y_step)
            ZsPC_ = predict(M, pcaDCA.one_hot(Zs_))
            y_i = mean(ZsPC_, dims=2)[:]
            Δ = y_f - y_i
            nΔ = norm(Δ)
            Δ ./= nΔ

            popfirst!(last_five)
            push!(last_five, nΔ)
            η = if std(last_five) < 0.1
                η*(1.0 + std(last_five))
            else
                η
            end


            
            # println("step ", i, " position: ", y_i[1:d].-y_f[1:d])

            ax1[1].plot(y_i[1], y_i[2], ".", ms = 1.0, c="k")
            
        end
        println("iteration m: ", m, " likelihood: ", likelihoods[m])

    end
    ax1[2].plot(round.(likelihoods, digits=2), c="k", ms = 5)
    subplots_adjust(hspace=0.35)  

    return fig


end

#In this version at each step we look for the sequence with the highest likelihood and use that as starting point for the next jump
function max_likelihood_path_search(net, var; msample::Int = 1000, d::Int= 2, seq_start=rand(axes(var.msa,2)), seq_arrival=rand(axes(var.msa,2)))

    M = fit(PCA, pcaDCA.one_hot(var.msa), maxoutdim=d)
    ZPC = predict(M, pcaDCA.one_hot(var.msa))

    close("all")
    fig, ax1 = subplots(2, 1, figsize=(6, 8), gridspec_kw=Dict(:height_ratios => [2, 1]))
    plot_density(ZPC[1,:], ZPC[2,:], ax1[1])
    Zss = []
    y_steps = []
    likelihood_path = []
    η = 0.1
    y_i = ZPC[:,seq_start][:]
    y_f = ZPC[:,seq_arrival][:]
    Δ = y_f - y_i
    nΔ = norm(Δ)
    Δ ./= nΔ # unit vector in the direction of the target
    i = 0
    #println("iteration: ", i, " norm: ", nΔ)
    ax1[1].plot(y_i[1],y_i[2],".", c="b")
    # ax1[1].plot(y_f[1],y_f[2],".", c="r")
    #ax[1].arrow(y_i[1],y_i[2], y_f[1]-y_i[1], y_f[2]-y_i[2], head_width=0.2, head_length=0.3, fc="blue", ec="black")
    last_five = rand(5)
    println("Starting at ", ZPC[1:d,seq_start], " and going to ", ZPC[1:d,seq_arrival])
    while nΔ > 0.2 && i < 250
        i += 1
        y_step = y_i + η*Δ
        push!(y_steps, y_step)
        Zs_ = sample(net, y_step, msample)
        ZsPC_ = predict(M, pcaDCA.one_hot(Zs_))
        l_ = [pcaDCA.likelihood(net, Zs_[:,s], ZsPC_[:,s]) for s in 1:msample]
        idx = argmax(l_)
        y_i = ZsPC_[1:2,idx]
        push!(likelihood_path, l_[idx])
        push!(Zss, Zs_[:,idx])
        Δ = y_f - y_i
        nΔ = norm(Δ)
        Δ ./= nΔ

        popfirst!(last_five)
        push!(last_five, nΔ)
        η = if std(last_five) < 0.1
            η*(1.0 + std(last_five))
        else
            η
        end


        # println("iteration: ", i, " norm: ", nΔ, " std: ", std(last_five))
        println("step ", i, " position: ", y_i[1:d].-y_f[1:d])

        ax1[1].plot(y_i[1], y_i[2], ".", c="k")
        # ax1[1].plot(y_step[1], y_step[2], ".", c="g")
    end
    ax1[1].plot(y_f[1],y_f[2],"x", c="r")
    ax1[2].set_title("likelihood, sum = $(round(sum(likelihood_path), digits=2))")
    ax1[2].plot(likelihood_path, label="likelihood")
    # savefig("pca_path.pdf")
    # plt.tight_layout()
    # plt.show()
    subplots_adjust(hspace=0.35)  

    return hcat(Zss...), y_steps, fig

end

function plot_frequency(Z)
    L,M = size(Z)
    q = maximum(Z)

    f1,_ = compute_freq(Z)
    F1 = reshape(f1, q-1, L)
    ff1 = vcat(F1, -sum(F1,dims=1).+1)
    close("all")
    matshow(ff1, cmap="gray_r")
    yticks(0:20,1:21)
    xticks(range(0,stop=L-1,step=5), Int.(range(1,stop=L,step=5)))
    colorbar()
    gcf()
end

# function max_likelihood_path(likelihood_grid, start, goal)

#     rows, cols = size(likelihood_grid)

#     # Initialize DP table
#     dp = fill(-Inf, rows, cols)
#     dp[start...] = likelihood_grid[start...]  # Start position takes its own energy

#     # Compute DP table
#     for i in 1:rows
#         for j in 1:cols
#             if (i, j) == start
#                 continue  # Skip the start position (already initialized)
#             end

#             # Check the two possible moves (from top or left)
#             best_prev = -Inf
#             if i > 1
#                 best_prev = max(best_prev, dp[i-1, j])  # Coming from above
#             end
#             if j > 1
#                 best_prev = max(best_prev, dp[i, j-1])  # Coming from left
#             end

#             dp[i, j] = best_prev + likelihood_grid[i, j]
#         end
#     end

#     # Backtrack to reconstruct the path
#     path = []
#     i, j = goal
#     while (i, j) != start
#         pushfirst!(path, (i, j))
#         if i > 1 && dp[i, j] == dp[i-1, j] + likelihood_grid[i, j]
#             i -= 1  # Move up
#         else
#             j -= 1  # Move left
#         end
#     end
#     pushfirst!(path, start)  # Add the start position

#     # close("all")
#     # matshow(likelihood_grid)
#     # colorbar()
#     # x = map(x->x[1]-1, path) 
#     # y = map(x->x[2]-1, path) 
#     # plot(x,y,".-", c="r")

#     return path, dp[goal...]

# end

function max_likelihood_path(likelihood_grid, start, goal)
    rows, cols = size(likelihood_grid)

    # Initialize DP table
    dp = fill(-Inf, rows, cols)
    dp[start...] = likelihood_grid[start...]  # Start position takes its own likelihood

    # First DP pass: Forward iteration (top-left to bottom-right)
    for i in 1:rows
        for j in 1:cols
            if (i, j) == start
                continue
            end
            
            # Best previous value
            best_prev = -Inf
            if i > 1
                best_prev = max(best_prev, dp[i-1, j])  # From top
            end
            if j > 1
                best_prev = max(best_prev, dp[i, j-1])  # From left
            end

            dp[i, j] = max(dp[i, j], best_prev + likelihood_grid[i, j])
        end
    end

    # Second DP pass: Backward iteration (bottom-right to top-left)
    for i in rows:-1:1
        for j in cols:-1:1
            if (i, j) == start
                continue
            end
            
            best_prev = dp[i, j]
            if i < rows
                best_prev = max(best_prev, dp[i+1, j] + likelihood_grid[i, j])  # From below
            end
            if j < cols
                best_prev = max(best_prev, dp[i, j+1] + likelihood_grid[i, j])  # From right
            end

            dp[i, j] = max(dp[i, j], best_prev)
        end
    end

    # Backtrack to reconstruct the path
    path = []
    i, j = goal
    while (i, j) != start
        pushfirst!(path, (i, j))

        # Find the best previous step
        best_prev = -Inf
        prev_i, prev_j = nothing, nothing

        if i > 1 && dp[i, j] == dp[i-1, j] + likelihood_grid[i, j]
            best_prev, prev_i, prev_j = dp[i-1, j], i-1, j
        end
        if i < rows && dp[i, j] == dp[i+1, j] + likelihood_grid[i, j] && dp[i+1, j] > best_prev
            best_prev, prev_i, prev_j = dp[i+1, j], i+1, j
        end
        if j > 1 && dp[i, j] == dp[i, j-1] + likelihood_grid[i, j] && dp[i, j-1] > best_prev
            best_prev, prev_i, prev_j = dp[i, j-1], i, j-1
        end
        if j < cols && dp[i, j] == dp[i, j+1] + likelihood_grid[i, j] && dp[i, j+1] > best_prev
            best_prev, prev_i, prev_j = dp[i, j+1], i, j+1
        end

        if isnothing(prev_i) || isnothing(prev_j)
            error("Failed to backtrack, possible DP inconsistency at ($i, $j)")
        end

        i, j = prev_i, prev_j
    end

    pushfirst!(path, start)  # Add the start position

    return path, dp[goal...]
end



function compute_likelihood_grid(net, var; n=30)
    M = fit(PCA, pcaDCA.one_hot(var.msa), maxoutdim=var.d)
    ZPC = predict(M, pcaDCA.one_hot(var.msa))
    max_x = maximum(ZPC[1,:])
    max_y = maximum(ZPC[2,:])
    min_x = minimum(ZPC[1,:])
    min_y = minimum(ZPC[2,:])


    # m_x = [range(min_x,max_x,n);] .+ ([range(min_x,max_x,n);][2]-[range(min_x,max_x,n);][1])/2
    # m_y = [range(min_y,max_y,n);] .+ ([range(min_y,max_y,n);][2]-[range(min_y,max_y,n);][1])/2;
    m_x = LinRange(min_x, max_x, n)
    m_y = LinRange(min_y, max_y, n)

    grid_likelihood = zeros(length(m_x)-1, length(m_y)-1)
    for i in 1:n-1
        for j in 1:n-1
            Y = [m_x[i],m_y[j]]
            Zs_ = sample(net, Y, 500)
            Zs_PC = predict(M, pcaDCA.one_hot(Zs_))
            # grid_likelihood[end-i+1,j] = pcaDCA.likelihood(net, Zs_, Y)
            grid_likelihood[i,j] = pcaDCA.likelihood(net, Zs_, Zs_PC)
        end
    end
    close("all")
    pcolormesh(m_x, m_y, grid_likelihood, cmap="Reds", shading="flat")

    return m_x, m_y, grid_likelihood

end

function point_to_index(x, y, xmin, xmax, ymin, ymax, N)
    # Ensure inputs are arrays for uniform processing
    x_arr = isa(x, Number) ? [x] : x
    y_arr = isa(y, Number) ? [y] : y
    
    # Compute tile size
    dx = (xmax - xmin) / N
    dy = (ymax - ymin) / N

    # Compute indices
    i_arr = clamp.(floor.( (x_arr .- xmin) ./ dx) .+ 1, 1, N)
    j_arr = clamp.(floor.( (y_arr .- ymin) ./ dy) .+ 1, 1, N)

    # Return single values if input was single, otherwise return arrays
    return isa(x, Number) ? Int.((j_arr[1],i_arr[1])) : Int.((j_arr,i_arr))
end

function index_to_center(i, j, xmin, xmax, ymin, ymax, N)
    # Ensure inputs are arrays for uniform processing
    i_arr = isa(i, Number) ? [i] : i
    j_arr = isa(j, Number) ? [j] : j

    # Compute tile size
    dx = (xmax - xmin) / N
    dy = (ymax - ymin) / N

    # Compute center coordinates
    x_center = xmin .+ (i_arr .- 0.5) .* dx
    y_center = ymin .+ (j_arr .- 0.5) .* dy

    # Return single values if input was single, otherwise return arrays
    return isa(i, Number) ? (x_center[1], y_center[1]) : (x_center, y_center)
end

function find_and_plot_path(Z, net, var, idx_start, idx_goal, grid_likelihood; d = 2)
    M = fit(PCA, pcaDCA.one_hot(var.msa), maxoutdim=d)
    ZPC = predict(M, pcaDCA.one_hot(Z))

    max_x = maximum(ZPC[1,:])
    max_y = maximum(ZPC[2,:])
    min_x = minimum(ZPC[1,:])
    min_y = minimum(ZPC[2,:])

    n = size(grid_likelihood, 1)

    m_x = LinRange(min_x, max_x, n+1)
    m_y = LinRange(min_y, max_y, n+1)

    start = pcaDCA.point_to_index(ZPC[1:2,idx_start]..., min_x, max_x, min_y, max_y, n)
    goal = pcaDCA.point_to_index(ZPC[1:2,idx_goal]..., min_x, max_x, min_y, max_y, n)
    
    # return start, goal
    path, likelihood = shortest_path(-grid_likelihood, start, goal)

    ps = pcaDCA.index_to_point.(path,n)
    x = map(x->x[1]+1, ps) 
    y = map(x->x[2]+1, ps)
    paths = pcaDCA.index_to_center(x, y, min_x, max_x, min_y, max_y, n)
   

    Zss, y_steps, likelihood_path = simple_path_search_no_plot(net, var, d = d, seq_start=idx_start, seq_arrival=idx_goal)

    likelihood_path_model = zeros(length(paths[1]))
    y_steps_model = []
    for i in 1:length(paths[1])
        Zs_ = sample(net, [paths[1][i],paths[2][i]], 500)
        Zs_PC = predict(M, pcaDCA.one_hot(Zs_))
        likelihood_path_model[i] = pcaDCA.likelihood(net, Zs_, Zs_PC)
        push!(y_steps_model, mean(Zs_PC, dims=2)[1:2])
    end

    close("all")

    pcolormesh(m_x, m_y, grid_likelihood, cmap="Reds", shading="flat")
    colorbar()
    plot_density(ZPC[1,:], ZPC[2,:])
    plot(ZPC[1,idx_start], ZPC[2,idx_start], "x", c="r", label="start")
    plot(ZPC[1,idx_goal], ZPC[2,idx_goal], "v", c="b", label="goal")
    plot(paths[1], paths[2], ".-", c="r")
    [plot(y[1], y[2], ".", c="k") for y in y_steps]
    [plot(y[1], y[2], ".", c="b") for y in y_steps_model]

    L1 = round(-likelihood,digits=2)
    L2 = round(sum(likelihood_path),digits=2)
    L3 = round(sum(likelihood_path_model),digits=2)
    text(-5.0, 5.0, "L_shortest: $L1, L_shortest_sampled: $L3, L_model: $L2",;
     fontsize=12, color="black",
     bbox=Dict("facecolor" => "lightblue", "alpha" => 0.5, "edgecolor" => "black"))

    # title("L_shortest: $(round(-likelihood,digits=2)), L_model: $(round(sum(likelihood_path),digits=2))")
    legend()

    gcf()

end

# Function to convert a matrix into a SimpleWeightedGraph
function matrix_to_graph(matrix)
    rows, cols = size(matrix)
    g = SimpleWeightedDiGraph(rows * cols)
    
    index = (i, j) -> (i - 1) * cols + j  # Convert (i, j) to a graph node index
    
    for i in 1:rows
        for j in 1:cols
            node = index(i, j)
            
            for di in -1:1, dj in -1:1
                ni, nj = i + di, j + dj
                if 1 <= ni <= rows && 1 <= nj <= cols && (di != 0 || dj != 0)
                    neighbor = index(ni, nj)
                    weight = matrix[ni, nj]
                    # println(node," -> ",neighbor," with weight ",weight)
                    add_edge!(g, node, neighbor, weight)
                end
            end
        end
    end
    return g
end

# Function to find the shortest path in the matrix graph
function shortest_path(matrix, start::Tuple{Int,Int}, goal::Tuple{Int,Int})
    g = matrix_to_graph(matrix)
    rows, cols = size(matrix)
    index = (i, j) -> (i - 1) * cols + j
    
    start_idx = index(start...)
    goal_idx = index(goal...)
    
    dijkstra_state = dijkstra_shortest_paths(g, start_idx)
    path = enumerate_paths(dijkstra_state, goal_idx)

    tot_weights = 0.0
    for i in 1:(length(path) - 1)
        u, v = path[i], path[i+1]
        tot_weights += get_weight(g,u, v)
    end
    tot_weights
    
    return path, tot_weights
end


function index_to_point(idx, n)
    x = mod1(idx, n)
    y = div(idx - 1, n) + 1
    return x-1, y-1
end


function compute_likelihood_grid_ardca(net::ArDCA.ArNet, var::ArDCA.ArVar; n=30)
    M = fit(PCA, pcaDCA.one_hot(var.Z), maxoutdim=2)
    ZPC = predict(M, pcaDCA.one_hot(var.Z))
    max_x = maximum(ZPC[1,:])
    max_y = maximum(ZPC[2,:])
    min_x = minimum(ZPC[1,:])
    min_y = minimum(ZPC[2,:])


    # m_x = [range(min_x,max_x,n);] .+ ([range(min_x,max_x,n);][2]-[range(min_x,max_x,n);][1])/2
    # m_y = [range(min_y,max_y,n);] .+ ([range(min_y,max_y,n);][2]-[range(min_y,max_y,n);][1])/2;
    m_x = LinRange(min_x, max_x, n)
    m_y = LinRange(min_y, max_y, n)

    idxs = [Vector{Any}() for _ in 1:n-1, _ in 1:n-1]
    for i in 1:29
        idx_ = findall(x->m_x[i]<=x<m_x[i+1],ZPC[1,:])
        for j in 1:29
            idx__ = findall(x->m_y[j]<=x<m_y[j+1],ZPC[2,idx_])
            idxs[i,j] = idx_[idx__]
        end
    end

    grid_likelihood = zeros(length(m_x)-1, length(m_y)-1)

    for i in 1:n-1
        for j in 1:n-1
            if isempty(idxs[i,j])
                grid_likelihood[i,j] = -Inf
            else
                grid_likelihood[i,j] = mean([ArDCA.loglikelihood(var.Z[:,s], net) for s in idxs[i,j]])
            end
        end
    end
    close("all")
    pcolormesh(m_x, m_y, grid_likelihood, cmap="Reds", shading="flat")

    return m_x, m_y, grid_likelihood

end 

function compute_likelihood_grid_pcadca(net::pcaDCA.ArNet, var::pcaDCA.ArVar; n=30)
    M = fit(PCA, pcaDCA.one_hot(var.msa), maxoutdim=2)
    ZPC = predict(M, pcaDCA.one_hot(var.msa))
    max_x = maximum(ZPC[1,:])
    max_y = maximum(ZPC[2,:])
    min_x = minimum(ZPC[1,:])
    min_y = minimum(ZPC[2,:])


    # m_x = [range(min_x,max_x,n);] .+ ([range(min_x,max_x,n);][2]-[range(min_x,max_x,n);][1])/2
    # m_y = [range(min_y,max_y,n);] .+ ([range(min_y,max_y,n);][2]-[range(min_y,max_y,n);][1])/2;
    m_x = LinRange(min_x, max_x, n)
    m_y = LinRange(min_y, max_y, n)

    idxs = [Vector{Any}() for _ in 1:n-1, _ in 1:n-1]
    for i in 1:29
        idx_ = findall(x->m_x[i]<=x<m_x[i+1],ZPC[1,:])
        for j in 1:29
            idx__ = findall(x->m_y[j]<=x<m_y[j+1],ZPC[2,idx_])
            idxs[i,j] = idx_[idx__]
        end
    end

    grid_likelihood = zeros(length(m_x)-1, length(m_y)-1)

    for i in 1:n-1
        for j in 1:n-1
            if isempty(idxs[i,j])
                grid_likelihood[i,j] = -Inf
            else
                grid_likelihood[i,j] = mean([likelihood(net, var.msa[:,s], var.Y[:,s]) for s in idxs[i,j]])
            end
        end
    end
    close("all")
    pcolormesh(m_x, m_y, grid_likelihood, cmap="Reds", shading="flat")

    return m_x, m_y, grid_likelihood

end 