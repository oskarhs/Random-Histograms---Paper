using Integrals, FastGaussQuadrature

import MergeSorted: mergesorted

include("test_distributions.jl")

function hellinger_loss(d::ContinuousUnivariateDistribution, breaks_hist::AbstractVector{<:Real}, dens_hist::AbstractVector{<:Real})
    disc = mergesorted(discontinuities(d), breaks_hist) # union of discontinuities of f₀ and \hat{f} in ascending order
    # Now perform numerical quadrature piecewise over each continuity interval
    m = length(disc) - 1
    hell = cdf(d, disc[1]-10*eps())
    for j = 1:m
        bin_ind = searchsortedfirst(breaks_hist, disc[j]+10*eps()) - 1
        if bin_ind == 0 || bin_ind == (length(dens_hist)+1)
            dens = 0.0
        else
            dens = dens_hist[bin_ind]
        end
        ip = IntegralProblem((t,p) -> (sqrt(pdf(d, t)) - sqrt(dens))^2, disc[j]+10*eps(), disc[j+1]-10*eps())
        hell += solve(ip, GaussLegendre()).u
    end
    hell += 1.0 - cdf(d, disc[end]+10*eps())
    return sqrt(hell)
end

function l2_loss(d::ContinuousUnivariateDistribution, breaks_hist::AbstractVector{<:Real}, dens_hist::AbstractVector{<:Real})
    disc = mergesorted(discontinuities(d), breaks_hist)
    supp = support(d)
    # Now perform numerical quadrature piecewise over each continuity interval
    m = length(disc) - 1
    ip = IntegralProblem((t,p) -> pdf(d, t)^2, supp.lb, disc[1]-10*eps())
    l2 = solve(ip, GaussLegendre()).u
    for j = 1:m
        bin_ind = searchsortedfirst(breaks_hist, disc[j]+10*eps()) - 1
        if bin_ind == 0 || bin_ind == (length(dens_hist)+1)
            dens = 0.0
        else
            dens = dens_hist[bin_ind]
        end
        ip = IntegralProblem((t,p) -> (pdf(d, t) - dens)^2, disc[j]+10*eps(), disc[j+1]-10*eps())
        l2 += solve(ip, GaussLegendre()).u
    end
    ip = IntegralProblem((t,p) -> pdf(d, t)^2, disc[end]+10*eps(), supp.ub)
    l2 += solve(ip, GaussLegendre()).u
    return sqrt(l2)
end


function peak_id_loss(d::ContinuousUnivariateDistribution, breaks_hist::AbstractVector{<:Real}, dens_hist::AbstractVector{<:Real};
                      verbose::Bool=false)
    # Identify locations of modes in the histogram density
    # First combine bins that have the same density (only needed for regular histograms)
    true_modes = modes(d)
    delta = pid_tolerance(d)
    xmin = breaks_hist[1]
    xmax = breaks_hist[end]

    k = length(dens_hist)
    dens_hist1 = [dens_hist[1]]
    breaks_hist1 = [breaks_hist[1]]
    for j in 1:(k-1)
        if !(isapprox(dens_hist[j], dens_hist[j+1], atol=100*eps(), rtol=0.0))
            push!(dens_hist1, dens_hist[j+1])
            push!(breaks_hist1, breaks_hist[j+1])
            #println("$j, $(dens_hist[j])")
        end
    end
    push!(breaks_hist1, breaks_hist[end])

    k = length(dens_hist1)

    #println(breaks_hist1)
    #println(dens_hist1)
    mids = zeros(k)
    for j = 1:k
        mids[j] = 0.5*(breaks_hist1[j]+breaks_hist1[j+1])
    end

    # Now identify modes of the histogram density
    hist_modes = Float64[]
    mode_val = Float64[]
    if k == 1
        push!(hist_modes, 0.5*(breaks_hist1[1]+breaks_hist1[2]))
        push!(mode_val, dens_hist1[1])
    else
        if dens_hist1[1] > dens_hist1[2]
            push!(hist_modes, 0.5*(breaks_hist1[1]+breaks_hist1[2]))
            push!(mode_val, dens_hist1[1])
        end
        for j in 2:(k-1)
            if dens_hist1[j] > dens_hist1[j-1] && dens_hist1[j] > dens_hist1[j+1]
                push!(hist_modes, 0.5*(breaks_hist1[j]+breaks_hist1[j+1]))
                push!(mode_val, dens_hist1[j])
            end
        end
        if dens_hist1[end] > dens_hist1[end-1]
            push!(hist_modes, 0.5*(breaks_hist1[end-1]+breaks_hist1[end]))
            push!(mode_val, dens_hist1[end])
        end
    end

    # Finally, see if the modes of the histogram match any of those from the true density
    C = 0 # number of correctly identified modes
    for j in eachindex(true_modes)
        if minimum(abs.(true_modes[j] .- hist_modes)) < delta[j]
            C = C + 1
        end
    end
    if verbose
        println("Not identified: $(length(true_modes) - C), Spurious: $(length(hist_modes) - C)")
    end
    return (length(true_modes) - C) + (length(hist_modes) - C), hist_modes, mode_val, (length(hist_modes) - C)
end
