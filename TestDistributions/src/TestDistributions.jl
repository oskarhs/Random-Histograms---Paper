module TestDistributions

# Common methods
export rand, pdf, cdf, support, peaks, pid_tolerance, discontinuities, peaks

# Supported distributions
export Normal, Laplace, LogNormal, Beta, Chisq, Uniform, Claw, SkewedBimodal, AsymmetricClaw, Sawtooth,
       SymTriangularDist, Marronite, Harp, TrimodalUniform, TenBinRegularHist, TenBinIrregularHist

# Plotting convenience functions
export plot_all_test_distributions, plot_test_distribution

using Random, Distributions, Plots, StatsBase

import Base: rand
import Distributions: pdf, cdf, support

include("distributions.jl")

const SUPPORTED_DISTRIBUTIONS = """
        - `Normal`, `Laplace`, `LogNormal`, `Beta`, `Chisq`, `Uniform`, `Claw`, `SkewedBimodal`, `AsymmetricClaw`, `SawTooth`,
        `SymTriangularDist`, `Marronite`, `Harp`, `TrimodalUniform`, `TenBinRegularHist`, `TenBinIrregularHist`.
"""

end # module TestDistributions
