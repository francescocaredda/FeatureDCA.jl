module pcaDCA

import DCAUtils: read_fasta_alignment,remove_duplicate_sequences,compute_weights, compute_weighted_frequencies
import Printf: @printf
import Distributions: wsample 
import Base.Threads: @spawn, @threads
import LoopVectorization: @turbo, @avx
using FastaIO
using ExtractMacro
using Statistics, LinearAlgebra, Tullio
using NLopt
using MultivariateStats

export ArVar, ArNet, trainer, sample


include("types.jl")
include("utils.jl")

include("autoregressive.jl")

end 
