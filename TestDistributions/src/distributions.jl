"""
    discontinuities(d::ContinuousUnivariateDistribution)

Return the points at which the probability density function of `d` is discontinuous, in increasing order.
"""
discontinuities(d::ContinuousUnivariateDistribution)

"""
    peaks(d::ContinuousUnivariateDistribution)

Return the modes/peaks of a density in increasing order.

Similar to Distributions.modes, but differs slightly in that infinite peaks are returned by this function and not the latter. 
For the exact mathematical definition of a mode used here, see Simensen et al. (2025).

References

* Simensen, O. H., Christensen, D. & Hjort, N. L. (2025). Random irregular histograms. *arXiv preprint*. doi: [10.48550/ARXIV.2505.22034](https://doi.org/10.48550/ARXIV.2505.22034)
"""
peaks(d::ContinuousUnivariateDistribution)

"""
    pid_tolerance(d::ContinuousUnivariateDistribution)

Compute the tolerance vector for the PID loss of the distribution `d`. Each entry in the returned vector corresponds to the same entry in `peaks(d)`.

This generic method provides no fallback, must be overridden for each distribution. For details on the PID loss, and a proposal for determining the PID tolerance vector for well-separated modes, see Simensen et al. (2025).

References

* Simensen, O. H., Christensen, D. & Hjort, N. L. (2025). Random irregular histograms. *arXiv preprint*. doi: [10.48550/ARXIV.2505.22034](https://doi.org/10.48550/ARXIV.2505.22034)
"""
pid_tolerance(d::ContinuousUnivariateDistribution)

# Functions to get the mode(s) of a distribution
function peaks(d::Normal)
    return [mean(d)]
end
function pid_tolerance(d::Normal)
    mu, sigma = params(d)
    if (mu, sigma) == (0.0, 1.0)
        return [0.645]
    else
        throw(ArgumentError("pid_tolerance is not implemented for Normal($mu, $sigma)."))
    end
end
function peaks(d::Laplace)
    return [mean(d)]
end
function pid_tolerance(d::Laplace)
    mu, sigma = params(d)
    if (mu, sigma) == (0.0, 1.0)
        return [0.404]
    else
        throw(ArgumentError("pid_tolerance is not implemented for Laplace($mu, $sigma)."))
    end
end
function peaks(d::LogNormal)
    mu, sigma = params(d)
    return [exp(mu - sigma^2)]
end
function pid_tolerance(d::LogNormal)
    mu, sigma = params(d)
    if (mu, sigma) == (0.0, 1.0)
        return [0.1665]
    else
        throw(ArgumentError("pid_tolerance is not implemented for LogNormal($mu, $sigma)."))
    end
end
function peaks(d::Chisq)
    return [maximum([params(d)[1]-2, 0.0])]
end
function pid_tolerance(d::Chisq)
    v = params(d)[1]
    if v == 1.0
        return [quantile(d, 0.1)]
    else
        throw(ArgumentError("pid_tolerance is not implemented for Chisq($v)."))
    end
end
function peaks(d::Beta) # NB! This differs from the definition used by Distributions.jl, where a mode has to be finite
    a, b = params(d)
    if a > 1.0 && b > 1.0
        return [(a-1)/(a+b-2)]
    elseif a < 1.0 && b < 1.0
        return [0.0, 1.0]
    elseif a <= 1.0 && b > 1.0
        return [0.0]
    elseif a > 1.0 && b <= 1.0
        return [1.0]
    end
end
function pid_tolerance(d::Beta)
    a, b = params(d)
    if a == 0.5 && b == 0.5
        return [quantile(d, 0.1), quantile(d, 0.9)]
    else
        throw(ArgumentError("pid_tolerance is not implemented for Beta($a, $b)."))
    end
end
function peaks(d::SymTriangularDist)
    return [params(d)[1]]
end
function pid_tolerance(d::SymTriangularDist)
    mu, sigma = params(d)
    if (mu, sigma) == (0.0, 1.0)
        return [0.286]
    else
        throw(ArgumentError("pid_tolerance is not implemented for SymTriangularDist($mu, $sigma)."))
    end
end
function peaks(d::Uniform)
    return [mean(params(d))]
end
function pid_tolerance(d::Uniform)
    return [0.2775]
end
function peaks(d::Cauchy)
    return [location(d)]
end
function peaks(d::TDist)
    return [0.0]
end
function peaks(d::Exponential)
    return [0.0]
end
function discontinuities(d::Normal)
    return Float64[]
end
function discontinuities(d::Laplace)
    return Float64[]
end
function discontinuities(d::LogNormal)
    return Float64[]
end
function discontinuities(d::Beta)
    a, b = params(d)
    if a ≤ 1.0 && b ≤ 1.0
        disc = Float64[0.0, 1.0]
    elseif a ≤ 1.0
        disc = Float64[0.0]
    elseif b ≤ 1.0
        disc = Float64[1.0]
    else
        disc = Float64[]
    end
    return disc
end
function discontinuities(d::SymTriangularDist)
    return Float64[]
end
function discontinuities(d::Chisq)
    return Float64[0.0]
end
function discontinuities(d::Uniform)
    a, b = params(d)
    return Float64[a, b]
end

"""
    Claw <: ContinuousUnivariateDistribution

Density 10 from Marron and Wand (1992).

References

* Marron, J. S. and Wand, M. P. Exact mean integrated squared error. *The Annals of Statistics*, **20**, 712-736. doi: [10.0.4.190/aos/1176348653](https://doi.org/10.0.4.190/aos/1176348653)
"""
struct Claw <: ContinuousUnivariateDistribution end # Density 10 from Marron and Wand (1992) [https://doi.org/10.0.4.190/aos/1176348653]
function pdf(d::Claw, x::Real)
    val = 0.5*pdf(Normal(0.0, 1.0), x)
    for j in -2:2
        val = val + 0.1 * pdf(Normal(0.5*j, 0.1), x)
    end
    return val
end
function cdf(d::Claw, x::Real)
    val = 0.5*cdf(Normal(0.0, 1.0), x)
    for j in -2:2
        val = val + 0.1 * cdf(Normal(0.5*j, 0.1), x)
    end
    return val
end
function rand(rng::AbstractRNG, d::Claw)
    u = rand(rng)
    if u < 0.5
        x = rand(rng, Normal(0.0, 1.0))
    elseif u < 0.6
        x = rand(rng, Normal(-1.0, 0.1))
    elseif u < 0.7
        x = rand(rng, Normal(-0.5, 0.1))
    elseif u < 0.8
        x = rand(rng, Normal(0.0, 0.1))
    elseif u < 0.9
        x = rand(rng, Normal(0.5, 0.1))
    else
        x = rand(rng, Normal(1.0, 0.1))
    end
    return x
end
function peaks(d::Claw)
    return Float64[-0.9969638, -0.4978001, 0.0, 0.4978001, 0.9969638]
end
function pid_tolerance(d::Claw)
    return [0.08, 0.0855, 0.0885, 0.0855, 0.08]
end
function discontinuities(d::Claw)
    return Float64[]
end
function support(d::Claw)
    return RealInterval{Float64}(-Inf, Inf)
end


"""
    SkewedBimodal <: ContinuousUnivariateDistribution

Density 8 from Marron and Wand (1992).

References

* Marron, J. S. and Wand, M. P. Exact mean integrated squared error. ^*The Annals of Statistics*, **20**, 712-736. doi: [10.0.4.190/aos/1176348653](https://doi.org/10.0.4.190/aos/1176348653)
"""
struct SkewedBimodal <: ContinuousUnivariateDistribution end
function pdf(d::SkewedBimodal, x::Real)
    val = 0.75*pdf(Normal(0.0, 1.0), x) + 0.25*pdf(Normal(1.5, 1.0/3.0), x)
    return val
end
function cdf(d::SkewedBimodal, x::Real)
    val = 0.75*cdf(Normal(0.0, 1.0), x) + 0.25*cdf(Normal(1.5, 1.0/3.0), x)
    return val
end
function rand(rng::AbstractRNG, d::SkewedBimodal)
    u = rand(rng)
    if u < 0.75
        x = rand(rng, Normal(0.0, 1.0))
    else
        x = rand(rng, Normal(1.5, 1.0/3.0))
    end
    return x
end
function peaks(d::SkewedBimodal)
    return Float64[0.0005446896, 1.4425250000]
end
function pid_tolerance(d::SkewedBimodal)
    return [0.744, 0.289]
end
function discontinuities(d::SkewedBimodal)
    return Float64[]
end
function support(d::SkewedBimodal)
    return RealInterval{Float64}(-Inf, Inf)
end


"""
    SmoothComb <: ContinuousUnivariateDistribution

Density 14 from Marron and Wand (1992).

References

* Marron, J. S. and Wand, M. P. Exact mean integrated squared error. *The Annals of Statistics*, **20**, 712-736. doi: [10.0.4.190/aos/1176348653](https://doi.org/10.0.4.190/aos/1176348653)
"""
struct SmoothComb <: ContinuousUnivariateDistribution end
function pdf(d::SmoothComb, x::Real)
    val = 32.0 / 63.0 * pdf(Normal(-31.0/21.0, 32.0/63.0), x) + 
        16.0 / 63.0 * pdf(Normal(17.0/21.0, 16.0/63.0), x) + 
        8.0 / 63.0 * pdf(Normal(41.0/21.0, 8.0/63.0), x) + 
        4.0 / 63.0 * pdf(Normal(53.0/21.0, 4.0/63.0), x) + 
        2.0 / 63.0 * pdf(Normal(59.0/21.0, 2.0/63.0), x) + 
        1.0 / 63.0 * pdf(Normal(62.0/21.0, 1.0/63.0), x)
    return val
end
function cdf(d::SmoothComb, x::Real)
    val = 32.0 / 63.0 * cdf(Normal(-31.0/21.0, 32.0/63.0), x) + 
        16.0 / 63.0 * cdf(Normal(17.0/21.0, 16.0/63.0), x) + 
        8.0 / 63.0 * cdf(Normal(41.0/21.0, 8.0/63.0), x) + 
        4.0 / 63.0 * cdf(Normal(53.0/21.0, 4.0/63.0), x) + 
        2.0 / 63.0 * cdf(Normal(59.0/21.0, 2.0/63.0), x) + 
        1.0 / 63.0 * cdf(Normal(62.0/21.0, 1.0/63.0), x)
    return val
end
function rand(rng::AbstractRNG, d::SmoothComb)
    u = rand(rng)
    if u < 32.0/63.0
        x = rand(rng, Normal(-31.0/21.0, 32.0/63.0))
    elseif u < 48.0/63.0
        x = rand(rng, Normal(17.0/21.0, 16.0/63.0))
    elseif u < 56.0/63.0
        x = rand(rng, Normal(41.0/21.0, 8.0/63.0))
    elseif u < 60.0/63.0
        x = rand(rng, Normal(53.0/21.0, 4.0/63.0))
    elseif u < 62.0/63.0
        x = rand(rng, Normal(59.0/21.0, 2.0/63.0))
    else
        x = rand(rng, Normal(62.0/21.0, 1.0/63.0))
    end
    return x
end
function peaks(d::SmoothComb)
    return Float64[-1.4761910, 0.8095009, 1.9523700, 2.5238040, 2.8095210, 2.9523800]
end

"""
    Sawtooth <: ContinuousUnivariateDistribution

Density 11 from Rozenholc et al. (2010).

References

* Rozenholc, Y., Mildenberger, T., & Gather, U. (2010). Combining regular and irregular histograms by penalized likelihood. *Computational Statistics & Data Analysis*, **54**, 3313–3323. doi: [10.1016/j.csda.2010.04.021](https://doi.org/10.1016/j.csda.2010.04.021)
"""
struct Sawtooth <: ContinuousUnivariateDistribution end 
function pdf(d::Sawtooth, x::Real)
    val = 0.0
    for j in -9:2:9
        val = val + 1.0/10.0 * pdf(SymTriangularDist(j, 1.0), x)
    end
    return val
end
function cdf(d::Sawtooth, x::Real)
    val = 0.0
    for j in -9:2:9
        val = val + 1.0/10.0 * cdf(SymTriangularDist(j, 1.0), x)
    end
    return val
end
function rand(rng::AbstractRNG, d::Sawtooth)
    N = 2.0 * rand(rng, DiscreteUniform(-4, 5)) - 1
    x = N + rand(rng, SymTriangularDist(0.0, 1.0))
    return x
end
function peaks(d::Sawtooth)
    return Float64[-9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0]
end
function pid_tolerance(d::Sawtooth)
    return [0.28574999999997736, 0.28574999999997736, 0.28574999999997736, 0.28574999999997736, 0.28574999999997736, 0.28574999999997736, 0.28574999999997736, 0.28574999999997736, 0.28574999999997736, 0.28574999999997736]
end
function discontinuities(d::Sawtooth)
    return Float64[]
end
function support(d::Sawtooth)
    return RealInterval{Float64}(-10.0, 10.0)
end

struct Rocket <: ContinuousUnivariateDistribution end
function pdf(d::Rocket, x::Real)
    y = abs(x)
    if y > π/1.3+0.8
        return 0.0
    end
    Z_c = 1.0 - cos(1.0) + 5.0*sin(1.0)
    Z_b = 1.0 + cos(0.26)
    if y < 1
        val = sin(y) + 5.0*cos(y)
    else
        val = 1.3*sin(1.3*(y-0.8))
    end
    val = 0.5 * val / (Z_c + Z_b)
    return val
end
function rand(rng::AbstractRNG, d::Rocket)
    Z_c = 1.0 - cos(1.0) + 5*sin(1.0)
    Z_b = 1.0 + cos(0.26)
    u = rand(rng)
    if u < 0.5
        s = -1
        u = 2.0 * u
    else
        s = 1
        u = 2.0 * (1.0 - u)
    end
    if u < Z_b / (Z_c + Z_b)
        x = 0.8 .+ 1.0 ./ 1.3 .* acos.(cos.(0.26) .- Z_b*rand(rng))
    else
        # Rejection sampling 
        not_acc = true
        while not_acc
            x = rand(rng)
            prob = ( abs(sin(x)) + 5.0*cos(x) ) / (1.093 * Z_c)
            u1 = rand(rng)
            if u1 < prob
                not_acc = false
            end
        end
    end
    return s * x
end

"""
    Marronite <: ContinuousUnivariateDistribution

Density 5 from Rozenholc et al. (2010).

References

* Rozenholc, Y., Mildenberger, T., & Gather, U. (2010). Combining regular and irregular histograms by penalized likelihood. *Computational Statistics & Data Analysis*, **54**, 3313–3323. doi: [10.1016/j.csda.2010.04.021](https://doi.org/10.1016/j.csda.2010.04.021)
"""
struct Marronite <: ContinuousUnivariateDistribution end
function pdf(d::Marronite, x::Real)
    val = 2.0/3.0 * pdf(Normal(0.0,1.0), x) + 1.0/3.0 * pdf(Normal(-20.0, 0.25), x)
    return val
end
function cdf(d::Marronite, x::Real)
    val = 2.0/3.0 * cdf(Normal(0.0,1.0), x) + 1.0/3.0 * cdf(Normal(-20.0, 0.25), x)
    return val
end
function rand(rng::AbstractRNG, d::Marronite)
    u = rand(rng)
    if u < 2.0/3.0
        x = rand(rng, Normal(0.0, 1.0))
    else
        x = rand(rng, Normal(-20.0, 0.25))
    end
    return x
end
function peaks(d::Marronite)
    return Float64[-20.0, 0.0]
end
function pid_tolerance(d::Marronite)
    return [0.1612, 0.6447]
end
function discontinuities(d::Marronite)
    return Float64[]
end
function support(d::Marronite)
    return RealInterval{Float64}(-Inf, Inf)
end


"""
    TrimodalUniform <: ContinuousUnivariateDistribution

Density 10 from Rozenholc et al. (2010).

References

* Rozenholc, Y., Mildenberger, T., & Gather, U. (2010). Combining regular and irregular histograms by penalized likelihood. *Computational Statistics & Data Analysis*, **54**, 3313–3323. doi: [10.1016/j.csda.2010.04.021](https://doi.org/10.1016/j.csda.2010.04.021)
"""
struct TrimodalUniform <: ContinuousUnivariateDistribution end
function pdf(d::TrimodalUniform, x::Real)
    if abs(x) < 1
        val = 0.5 * 0.5
    elseif 20 < abs(x) < 20.1
        val = 0.25 * 10.0
    else 
        val = 0.0
    end
    return val
end
function cdf(d::TrimodalUniform, x::Real)
    if x <= -20.0
        val = 0.25*cdf(Uniform(-20.1, -20.0), x)
    elseif x < 20.0
        val = 0.25 + 0.5*cdf(Uniform(-1.0, 1.0), x)
    elseif x <= 20.1
        val = 0.75 + 0.25*cdf(Uniform(20.0, 20.1), x)
    else 
        val = 1.0
    end
    return val
end
function rand(rng::AbstractRNG, d::TrimodalUniform)
    u = rand(rng)
    if u < 0.5
        x = rand(rng, Uniform(-1.0, 1.0))
    elseif u < 0.75
        x = rand(rng, Uniform(20, 20.1))
    else
        x = rand(rng, Uniform(-20.1, -20.0))
    end
    return x
end
function peaks(d::TrimodalUniform)
    return Float64[-20.05, 0.00, 20.05]
end
function pid_tolerance(d::TrimodalUniform)
    return [0.02775, 0.55485, 0.02775]
end
function discontinuities(d::TrimodalUniform)
    return Float64[-20.1, -20.0, -1.0, 1.0, 20.0, 20.1]
end
function support(d::TrimodalUniform)
    return RealInterval{Float64}(-20.1, 20.1)
end


"""
    TenBinRegularHist <: ContinuousUnivariateDistribution

Density 15 from Rozenholc et al. (2010).

References

* Rozenholc, Y., Mildenberger, T., & Gather, U. (2010). Combining regular and irregular histograms by penalized likelihood. *Computational Statistics & Data Analysis*, **54**, 3313–3323. doi: [10.1016/j.csda.2010.04.021](https://doi.org/10.1016/j.csda.2010.04.021)
"""
struct TenBinRegularHist <: ContinuousUnivariateDistribution end
function pdf(d::TenBinRegularHist, x::Real)
    mixtureprob = [0.01, 0.18, 0.16, 0.07, 0.06, 0.01, 0.06, 0.37, 0.06, 0.02]
    breakpoints = LinRange(0.0, 1.0, 11)
    val = 0.0
    for i in eachindex(mixtureprob)
        val = val + mixtureprob[i] * pdf(Uniform(breakpoints[i], breakpoints[i+1]), x)
    end
    return val
end
function cdf(d::TenBinRegularHist, x::Real)
    mixtureprob = [0.01, 0.18, 0.16, 0.07, 0.06, 0.01, 0.06, 0.37, 0.06, 0.02]
    breakpoints = LinRange(0.0, 1.0, 11)
    val = 0.0
    for i in eachindex(mixtureprob)
        val = val + mixtureprob[i] * cdf(Uniform(breakpoints[i], breakpoints[i+1]), x)
    end
    return val
end
function rand(rng::AbstractRNG, d::TenBinRegularHist)
    mixtureprob = ProbabilityWeights([0.01, 0.18, 0.16, 0.07, 0.06, 0.01, 0.06, 0.37, 0.06, 0.02])
    breakpoints = LinRange(0.0, 1.0, 11)
    j = StatsBase.sample(rng, 1:10, mixtureprob) # Sample index of mixture
    x = rand(rng, Uniform(breakpoints[j], breakpoints[j+1]))
    return x
end
function peaks(d::TenBinRegularHist)
    return Float64[0.15, 0.75]
end
function pid_tolerance(d::TenBinRegularHist)
    return Float64[0.03175, 0.0285]
end
function discontinuities(d::TenBinRegularHist)
    return collect(LinRange(0.0, 1.0, 11))
end
function support(d::TenBinRegularHist)
    return RealInterval{Float64}(0.0, 1.0)
end

"""
    TenBinIrregularHist <: ContinuousUnivariateDistribution

Density 16 from Rozenholc et al. (2010).

References

* Rozenholc, Y., Mildenberger, T., & Gather, U. (2010). Combining regular and irregular histograms by penalized likelihood. *Computational Statistics & Data Analysis*, **54**, 3313–3323. doi: [10.1016/j.csda.2010.04.021](https://doi.org/10.1016/j.csda.2010.04.021)
"""
struct TenBinIrregularHist <: ContinuousUnivariateDistribution end
function pdf(d::TenBinIrregularHist, x::Real)
    mixtureprob = [0.01, 0.18, 0.16, 0.07, 0.06, 0.01, 0.06, 0.37, 0.06, 0.02]
    breakpoints = [0, 0.02, 0.07, 0.14, 0.44, 0.53, 0.56, 0.67, 0.77, 0.91, 1.0]
    val = 0.0
    for i in eachindex(mixtureprob)
        val = val + mixtureprob[i] * pdf(Uniform(breakpoints[i], breakpoints[i+1]), x)
    end
    return val
end
function cdf(d::TenBinIrregularHist, x::Real)
    mixtureprob = [0.01, 0.18, 0.16, 0.07, 0.06, 0.01, 0.06, 0.37, 0.06, 0.02]
    breakpoints = [0, 0.02, 0.07, 0.14, 0.44, 0.53, 0.56, 0.67, 0.77, 0.91, 1.0]
    val = 0.0
    for i in eachindex(mixtureprob)
        val = val + mixtureprob[i] * cdf(Uniform(breakpoints[i], breakpoints[i+1]), x)
    end
    return val
end
function rand(rng::AbstractRNG, d::TenBinIrregularHist)
    mixtureprob = ProbabilityWeights([0.01, 0.18, 0.16, 0.07, 0.06, 0.01, 0.06, 0.37, 0.06, 0.02])
    breakpoints = [0.0, 0.02, 0.07, 0.14, 0.44, 0.53, 0.56, 0.67, 0.77, 0.91, 1.0]
    j = StatsBase.sample(rng, 1:10, mixtureprob) # Sample index of mixture
    x = rand(rng, Uniform(breakpoints[j], breakpoints[j+1]))
    return x
end
function peaks(d::TenBinIrregularHist)
    return [0.045, 0.485, 0.72]
end
function pid_tolerance(d::TenBinIrregularHist)
    return [0.01512, 0.02785, 0.0283]
end
function discontinuities(d::TenBinIrregularHist)
    return Float64[0.0, 0.02, 0.07, 0.14, 0.44, 0.53, 0.56, 0.67, 0.77, 0.91, 1.0]
end
function support(d::TenBinIrregularHist)
    return RealInterval{Float64}(0.0, 1.0)
end


"""
    StronglySkewed <: ContinuousUnivariateDistribution

Density 3 from Marron and Wand (1992).

References

* Marron, J. S. and Wand, M. P. Exact mean integrated squared error. *The Annals of Statistics*, **20**, 712-736. doi: [10.0.4.190/aos/1176348653](https://doi.org/10.0.4.190/aos/1176348653)
"""
struct StronglySkewed <: ContinuousUnivariateDistribution end # Density 3 in Marron and Wand (1992) [https://10.0.4.190/aos/1176348653]
function pdf(d::StronglySkewed, x::Real)
    dens = 0.0
    for j = 0:7
        dens = dens + 0.125 * pdf(Normal(3.0*((2.0/3.0)^j - 1), sqrt((2.0/3.0)^(2*j))), x)
    end
    return dens
end
function cdf(d::StronglySkewed, x::Real)
    dens = 0.0
    for j = 0:7
        dens = dens + 0.125 * cdf(Normal(3.0*((2.0/3.0)^j - 1), sqrt((2.0/3.0)^(2*j))), x)
    end
    return dens
end
function rand(rng::AbstractRNG, d::StronglySkewed)
    j = rand(rng, DiscreteUniform(0, 7))
    return rand(rng, Normal(3.0*((2.0/3.0)^j - 1), sqrt((2.0/3.0)^(2*j))))
end

struct SymmetricBimodal <: ContinuousUnivariateDistribution end
function pdf(d::SymmetricBimodal, x::Real)
    return 0.5 * pdf(Normal(-1.3, 0.9), x) + 0.5 * pdf(Normal(1.3, 0.9), x)
end
function cdf(d::SymmetricBimodal, x::Real)
    return 0.5 * cdf(Normal(-1.3, 0.9), x) + 0.5 * cdf(Normal(1.3, 0.9), x)
end
function rand(rng::AbstractRNG, d::SymmetricBimodal)
    u = rand(rng)
    if u < 0.5
        return rand(rng, Normal(-1.3, 0.9))
    else
        return rand(rng, Normal(1.3, 0.9))
    end
end
function peaks(d::SymmetricBimodal)
    return [-1.2544420578384188, 1.2544420578384188]
end


"""
    AsymmetricClaw <: ContinuousUnivariateDistribution

Density 12 from Marron and Wand (1992).

References

* Marron, J. S. and Wand, M. P. Exact mean integrated squared error. *The Annals of Statistics*, **20**, 712-736. doi: [10.0.4.190/aos/1176348653](https://doi.org/10.0.4.190/aos/1176348653)
"""
struct AsymmetricClaw <: ContinuousUnivariateDistribution end
function pdf(d::AsymmetricClaw, x::Real)
    dens = 0.5*pdf(Normal(0.0, 1.0), x)
    for l=-2:2
        dens = dens + 2.0^(1.0-l)/31.0 * pdf(Normal(l+0.5, 0.1*2.0^(-l)), x)
    end
    return dens
end
function cdf(d::AsymmetricClaw, x::Real)
    val = 0.5*cdf(Normal(0.0, 1.0), x)
    for l=-2:2
        val += 2.0^(1.0-l)/31.0 * cdf(Normal(l+0.5, 0.1*2.0^(-l)), x)
    end
    return val
end
function rand(rng::AbstractRNG, d::AsymmetricClaw)
    u = rand(rng)
    cumprob = [0.75806451612903225, 0.8870967741935484, 0.95161290322580644, 0.9838709677419355, 1.0]
    if u < 0.5
        return rand(rng, Normal(0.0 ,1.0))
    else
        for l=-2:2
            if u <= cumprob[l+3]
                return rand(rng,  Normal(l+0.5, 0.1*2.0^(-l)))
            end
        end
    end
end
function peaks(d::AsymmetricClaw)
    return [-1.435580504234139, -0.49720234625184384, 0.49659437764952674,
            1.4990555680283049, 2.499946789043634]
end
function pid_tolerance(d::AsymmetricClaw)
    return [0.36149999999996835, 0.23064999999998276, 0.09874999999999728,
            0.03810000000000058, 0.01649999999999996]
end
function discontinuities(d::AsymmetricClaw)
    return Float64[]
end
function support(d::AsymmetricClaw)
    return RealInterval{Float64}(-Inf, Inf)
end

struct TruncatedNormalMixture <: ContinuousUnivariateDistribution end
function pdf(d::TruncatedNormalMixture, x::Real)
    dens = (
        0.2*pdf(truncated(Normal(0.2, 0.1); lower=0.0, upper=0.4), x) +
        0.5*pdf(truncated(Normal(0.55, 0.1); lower=0.4, upper=0.7), x) +
        0.1*pdf(truncated(Normal(0.8, 0.1); lower=0.7, upper=0.9), x) +
        0.2*pdf(truncated(Normal(0.95, 0.1); lower=0.9, upper=1.0), x)
    )
    return dens
end
function rand(rng::AbstractRNG, d::TruncatedNormalMixture)
    u = rand(rng)
    if u < 0.2
        return rand(rng, truncated(Normal(0.2, 0.1); lower=0.0, upper=0.4))
    elseif u < 0.7
        return rand(rng, truncated(Normal(0.55, 0.1); lower=0.4, upper=0.7))
    elseif u < 0.8
        return rand(rng, truncated(Normal(0.8, 0.1); lower=0.7, upper=0.9))
    else
        return rand(rng, truncated(Normal(0.95, 0.1); lower=0.9, upper=1.0))
    end
end


"""
    SeparatedBimodal <: ContinuousUnivariateDistribution

Density 7 from Marron and Wand (1992).

References

* Marron, J. S. and Wand, M. P. Exact mean integrated squared error. *The Annals of Statistics*, **20**, 712-736. doi: [10.0.4.190/aos/1176348653](https://doi.org/10.0.4.190/aos/1176348653)
"""
struct SeparatedBimodal <: ContinuousUnivariateDistribution end
function pdf(d::SeparatedBimodal, x::Real)
    return 0.3 * pdf(Normal(-1.5, 0.5), x) + 0.7 * pdf(Normal(1.5, 0.5), x)
end
function rand(rng::AbstractRNG, d::SeparatedBimodal)
    u = rand(rng)
    if u < 0.3
        return rand(rng, Normal(-1.5, 0.5))
    else
        return rand(rng, Normal(1.5, 0.5))
    end
end


"""
    Outlier <: ContinuousUnivariateDistribution

Density 5 from Marron and Wand (1992).

References

* Marron, J. S. and Wand, M. P. Exact mean integrated squared error. *The Annals of Statistics*, **20**, 712-736. doi: [10.0.4.190/aos/1176348653](https://doi.org/10.0.4.190/aos/1176348653)
"""
struct Outlier <: ContinuousUnivariateDistribution end
function pdf(d::Outlier, x::Real)
    return 0.1*pdf(Normal(0.0, 1.0), x) + 0.9 * pdf(Normal(0.0, 0.1), x)
end
function cdf(d::Outlier, x::Real)
    return 0.1*cdf(Normal(0.0, 1.0), x) + 0.9 * cdf(Normal(0.0, 0.1), x)
end
function rand(rng::AbstractRNG, d::Outlier)
    u = rand(rng)
    if u < 0.1
        return rand(rng, Normal(0.0, 1.0))
    else
        return rand(rng, Normal(0.0, 0.1))
    end
end
function peaks(d::Outlier)
    return Float64[0.0]
end


"""
    TenNormalMixture <: ContinuousUnivariateDistribution

Density 16 from Davies et al. (2009).

References

* Davies, L., Gather, U., Nordman, D. & Weinert, H. (2009). A comparison of automatic histogram constructions. *ESAIM: Probability and Statistics*, **13**, 181-196. doi: [10.1051/ps:2008005](https://doi.org/10.1051/ps:2008005)
"""
struct TenNormalMixture <: ContinuousUnivariateDistribution end
function pdf(d::TenNormalMixture, x::Real)
    dens = 0.0
    for i=1:10
        dens = dens + pdf(Normal(5*i-5, 1), x)
    end
    return 0.1 * dens
end
function cdf(d::TenNormalMixture, x::Real)
    cum = 0.0
    for i=1:10
        cum = cum + cdf(Normal(5*i-5, 1), x)
    end
    return 0.1 * cum
end
function rand(rng::AbstractRNG, d::TenNormalMixture)
    j = rand(rng, DiscreteUniform(1, 10))
    return rand(rng, Normal(5*j-5, 1))
end


"""
    Harp <: ContinuousUnivariateDistribution

Harp density from Li et al. (2020).

References

* Li, H., Munk, A., Sieling, H. and Walther, G. (2020). The essential histogram. *Biometrika*, **107**, 347-364. doi: [10.1093/biomet/asz081](https://doi.org/10.1093/biomet/asz081)
"""
struct Harp <: ContinuousUnivariateDistribution end
function pdf(d::Harp, x::Real)
    means = [0.0, 5.0, 15.0, 30.0, 60.0]
    sds = [0.5, 1.0, 2.0, 4.0, 8.0]
    dens = 0.0
    for j = 1:5
        dens = dens + 0.2 * pdf(Normal(means[j], sds[j]), x)
    end
    return dens
end
function cdf(d::Harp, x::Real)
    means = [0.0, 5.0, 15.0, 30.0, 60.0]
    sds = [0.5, 1.0, 2.0, 4.0, 8.0]
    cum = 0.0
    for j = 1:5
        cum += 0.2 * cdf(Normal(means[j], sds[j]), x)
    end
    return cum
end
function rand(rng::AbstractRNG, d::Harp)
    means = [0.0, 5.0, 15.0, 30.0, 60.0]
    sds = [0.5, 1.0, 2.0, 4.0, 8.0]
    j = rand(rng, DiscreteUniform(1, 5))
    return rand(rng, Normal(means[j], sds[j]))
end
function peaks(d::Harp)
    return Float64[0.0, 5.0, 15.001837158203125, 30.003319148997427, 60.000555419921874]
end
function pid_tolerance(d::Harp)
    return [0.32235, 0.6447, 1.2919, 2.58375, 5.1574]
end
function discontinuities(d::Harp)
    return Float64[]
end
function support(d::Harp)
    return RealInterval{Float64}(-Inf, Inf)
end


"""
    AsymmetricDoubleClaw <: ContinuousUnivariateDistribution

Density 13 from Marron and Wand (1992).

References

* Marron, J. S. and Wand, M. P. Exact mean integrated squared error. *The Annals of Statistics*, **20**, 712-736. doi: [10.0.4.190/aos/1176348653](https://doi.org/10.0.4.190/aos/1176348653)
"""
struct AsymmetricDoubleClaw <: ContinuousUnivariateDistribution end
function pdf(d::AsymmetricDoubleClaw, x::Real)
    dens = 0.46*pdf(Normal(-1.0, 2.0/3.0), x) + 0.46*pdf(Normal(1.0, 2.0/3.0), x)
    dens = dens + 1.0/300.0 * (pdf(Normal(-0.5, 0.01), x) + pdf(Normal(-1.0, 0.01), x) + pdf(Normal(-1.5, 0.01), x))
    dens = dens + 7.0/300.0 * (pdf(Normal(0.5, 0.07), x) + pdf(Normal(1.0, 0.07), x) + pdf(Normal(1.5, 0.07), x))
    return dens
end
function cdf(d::AsymmetricDoubleClaw, x::Real)
    cum = 0.46*cdf(Normal(-1.0, 2.0/3.0), x) + 0.46*cdf(Normal(1.0, 2.0/3.0), x)
    cum += 1.0/300.0 * (cdf(Normal(-0.5, 0.01), x) + cdf(Normal(-1.0, 0.01), x) + cdf(Normal(-1.5, 0.01), x))
    cum += 7.0/300.0 * (cdf(Normal(0.5, 0.07), x) + cdf(Normal(1.0, 0.07), x) + cdf(Normal(1.5, 0.07), x))
    return cum
end
function rand(rng::AbstractRNG, d::AsymmetricDoubleClaw)
    mixtureprob = [0.46, 0.46, 1.0/300.0, 1.0/300.0, 1.0/300.0, 7.0/300.0, 7.0/300.0, 7.0/300.0]
    mix_mean = [-1.0, 1.0, -0.5, -1.0, -1.5, 0.5, 1.0, 1.5]
    mix_std = [2.0/3.0, 2.0/3.0, 0.01, 0.01, 0.01, 0.07, 0.07, 0.07]
    j = wsample(1:8, mixtureprob)
    return rand(rng, Normal(mix_mean[j], mix_std[j]))
end
function peaks(d::AsymmetricDoubleClaw)
    return Float64[
        -0.500120209315021, -0.999989656351693, -1.499823186172658,
        0.5059088699390483, 0.999503262096811, 1.4913348060631115
    ]
end

# Return an array containing the test densities defined above
function get_test_distributions()
    test_distributions = [
        Normal(), Uniform(), Chisq(1.0), LogNormal(), 
        Laplace(), Beta(0.5, 0.5), Claw(), SkewedBimodal(),
        AsymmetricClaw(), Sawtooth(), SymTriangularDist(), Marronite(),
        Harp(), TrimodalUniform(), TenBinRegularHist(), TenBinIrregularHist()
        ]
    return test_distributions
end

function get_tuning_distributions()
    tuning_distributions = [
        TDist(3.0), Gamma(3.0, 3.0), Beta(3.0, 3.0)
    ]
    return tuning_distributions
end

"""
    plot_all_test_distributions()

Plot the probability density function of all the supported test distributions on a 4x4 grid. Returns a plot object.
"""
function plot_all_test_distributions()
    test_distributions = get_test_distributions()
    dom = [
        LinRange(-3.5, 3.5, 5000), LinRange(-0.001, 1.001, 5000),
        LinRange(0.0, 3.0, 5000), LinRange(0.0, 10.0, 5000), 
        LinRange(-5.0, 5.0, 5000), LinRange(0.001, 1.0-0.001, 5000),
        LinRange(-3.5, 3.5, 5000), LinRange(-3.5, 3.5, 5000),
        LinRange(-3.5, 3.5, 5000), LinRange(-10.5, 10.5, 5000),
        LinRange(-1.001, 1.001, 5000), LinRange(-22.0, 3.5, 5000),
        LinRange(-5.0, 80.0, 5000), LinRange(-22.5, 22.5, 5000),
        LinRange(-0.001, 1.001, 5000), LinRange(-0.001, 1.001, 5000)
    ]
    x_lims = [
        [-3.5, 3.5], [-0.1, 1.1],
        [-0.3, 3.1], [-0.8, 10.0],
        [-5.0, 5.0], [-0.1, 1.1],
        [-3.5, 3.5], [-3.5, 3.5],
        [-3.5, 3.5], [-10.5, 10.5],
        [-1.2, 1.2], [-23.0, 3.5],
        [-5.0, 80.0], [-24.5, 24.5],
        [-0.1, 1.1], [-0.1, 1.1]
    ]
    x_ticks = [
        -3:1:3, 0.0:0.2:1.0, 0:1:3, 0:2:10, -4:2:4, 0.0:0.2:1.0, -3:1:3, -3:1:3,
        -3:1:3, -10:5:10, -1.0:0.5:1.0, -20:5:0, 0:20:80, -20:10:20, 0.0:0.2:1.0, 0.0:0.2:1.0
    ]
    titles = [
        "#1 Standard Normal", "#2 Uniform", "#3 Chisq(1)", "#4 Standard Lognormal",
        "#5 Standard Laplace", "#6 Beta(0.5, 0.5)", "#7 Claw",  "#8 Skewed Bimodal",
        "#9 Asymmetric Claw", "#10 Sawtooth", "#11 Triangle", "#12 Marronite",
        "#13 Harp", "#14 Trimodal Uniform", "#15 Regular Hist", "#16 Irregular Hist"
    ]
    ps = []
    
    for i in eachindex(dom)
        d = test_distributions[i]
        t = dom[i]
        p = plot(t, pdf.(d, t), title=titles[i], label="", ylims=(0.0, Inf),
                xtickfontsize=7, ytickfontsize=7, titlefontsize=10, color="blue")
        xlims!(p, x_lims[i]...)
        if i == 2
            ylims!(p, 0.0, 1.1)
        end
        xticks!(p, x_ticks[i])
        push!(ps, p)
    end
    pl = plot(ps..., layout=(4,4), size=(900, 900), margin = 0.5*Plots.cm, legend=false)
    return pl
    #savefig(pl, "figures/TestDistributions.pdf")
end


"""
    plot_test_distribution(d::ContinuousUnivariateDistribution)

Plot the probability density function of the test distribution `d`.
"""
function plot_test_distribution(d::ContinuousUnivariateDistribution)
    test_distributions = get_test_distributions()
    if !(d in get_test_distributions())
        throw(ArgumentError("Supplied distribution $d is not supported."))
    end
    dom = [
        LinRange(-3.5, 3.5, 5000), LinRange(-0.001, 1.001, 5000),
        LinRange(0.0, 3.0, 5000), LinRange(0.0, 10.0, 5000), 
        LinRange(-5.0, 5.0, 5000), LinRange(0.001, 1.0-0.001, 5000),
        LinRange(-3.5, 3.5, 5000), LinRange(-3.5, 3.5, 5000),
        LinRange(-3.5, 3.5, 5000), LinRange(-10.5, 10.5, 5000),
        LinRange(-1.001, 1.001, 5000), LinRange(-22.0, 3.5, 5000),
        LinRange(-5.0, 80.0, 5000), LinRange(-22.5, 22.5, 5000),
        LinRange(-0.001, 1.001, 5000), LinRange(-0.001, 1.001, 5000)
    ]
    x_lims = [
        [-3.5, 3.5], [-0.1, 1.1],
        [-0.3, 3.1], [-0.8, 10.0],
        [-5.0, 5.0], [-0.1, 1.1],
        [-3.5, 3.5], [-3.5, 3.5],
        [-3.5, 3.5], [-10.5, 10.5],
        [-1.2, 1.2], [-23.0, 3.5],
        [-5.0, 80.0], [-24.5, 24.5],
        [-0.1, 1.1], [-0.1, 1.1]
    ]
    titles = [
        "#1 Standard Normal", "#2 Uniform", "#3 Chisq(1)", "#4 Standard Lognormal",
        "#5 Standard Laplace", "#6 Beta(0.5, 0.5)", "#7 Claw",  "#8 Skewed Bimodal",
        "#9 Asymmetric Claw", "#10 Sawtooth", "#11 Triangle", "#12 Marronite",
        "#13 Harp", "#14 Trimodal Uniform", "#15 Regular Hist", "#16 Irregular Hist"
    ]
    p = plot()
    for i in eachindex(dom)
        if isequal(d, test_distributions[i])
            t = dom[i]
            plot!(p, t, pdf.(d, t), title=titles[i], label="", ylims=(0.0, Inf), titlefontsize=10, color="blue")
            xlims!(p, x_lims[i]...)
            if i == 2
                ylims!(p, 0.0, 1.1)
            end
        end
    end
    return p
end
#plot_test_distributions()