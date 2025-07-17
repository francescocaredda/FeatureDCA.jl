module FeatureDCA

import DCAUtils: read_fasta_alignment,remove_duplicate_sequences,compute_weights, compute_weighted_frequencies
import Printf: @printf
import Distributions: wsample 
import Base.Threads: @spawn, @threads
import LoopVectorization: @turbo, @avx
import DelimitedFiles: readdlm
import OptimalTransport: sinkhorn_divergence
import Distances: Euclidean, pairwise, Hamming
import MultivariateStats: fit, predict, PCA
using FastaIO
using ExtractMacro
using Statistics, LinearAlgebra, Tullio

using NLopt

export ArVar, ArNet, trainer, sample


include("types.jl")
include("utils.jl")
include("autoregressive.jl")
include("dca.jl")
include("test.jl")

end 
