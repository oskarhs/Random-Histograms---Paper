using AutoHist, Integrals, Random, Distributions, Plots
import SpecialFunctions.loggamma

function hell_loss(dens_hist, bin_edges, d)
    k = length(bin_edges) - 1

    ip = IntegralProblem((t,p) -> pdf(d, t), -Inf, bin_edges[1])
    hell = solve(ip, QuadGKJL()).u
    for j = 1:k
        ip = IntegralProblem((t,p) -> (sqrt(pdf(d, t)) - sqrt(dens_hist[j]))^2, bin_edges[j], bin_edges[j+1])
        hell = hell + solve(ip, QuadGKJL()).u
    end
    ip = IntegralProblem((t,p) -> pdf(d, t), bin_edges[k+1], Inf)
    hell = hell + solve(ip, QuadGKJL()).u
    return sqrt(hell)
end

function compute_risk_a(n, d, rng)
    B = 500

    as = [0.01, 0.1, 1.0, 10.0, 100.0]
    loss = Array{Float64}(undef, length(as), B)
    k_opt = Array{Float64}(undef, length(as), B)
    risk = Array{Float64}(undef, length(as))
    mean_k = Array{Float64}(undef, length(as))

    support_d = (support(d).lb, support(d).ub)

    for j in eachindex(as)
        for b = 1:B
            x = rand(rng, d, n)
            H = histogram_irregular(x; a = as[j], grid=:regular, support=support_d,
                alg = ifelse(n ≥ 500, GPDP(), DP()))
            loss[j,b] = hell_loss(H.density, H.breaks, d)
            k_opt[j,b] = length(H.density)
        end
        risk[j] = mean(loss[j,:])
        mean_k[j] = mean(k_opt[j,:])
    end
    return risk, mean_k
end

function compute_risk_k(n, d, rng)
    B = 500

    # Uniform, 1/k, 1/k^2 and Poisson priors
    logpriors = [k->0.0, k->-log(k), k->-2.0*log(k), k->-loggamma(k+1)]
    loss = Array{Float64}(undef, length(logpriors), B)
    risk = Array{Float64}(undef, length(logpriors))

    support_d = (support(d).lb, support(d).ub)

    for j in eachindex(logpriors)
        for b = 1:B
            x = rand(rng, d, n)
            H = histogram_irregular(x; a = 1.0, logprior=logpriors[j], grid=:regular, support=support_d)
            loss[j,b] = hell_loss(H.density, H.breaks, d)
        end
        risk[j] = mean(loss[j,:])
    end
    return risk
end


function plot_risks_k()
    ns = [100, 1000, 10000]
    dists = [Gamma(3.0, 3.0), Beta(3.0, 3.0), TDist(3.0)]
    risks = Array{Float64}(undef, length(ns), length(dists), 4)

    for i in eachindex(ns)
        for j in eachindex(dists)
            risks[i,j,:] = compute_risk_k(ns[i], dists[j], Xoshiro(1812))
        end
    end
    
    p = plot(xticks=(1:3, ["Gamma(3,3)", "Beta(3, 3)", "t(3)"]),
            ylabel="log relative risk", legend=:outerright, fg_legend=:transparent)
    scatter!(p, [NaN], [NaN], label="pₙ(k)∝1/k", markershape=:dtriangle, color="lightgrey", msa = 1.0, msc="black", msw=1.5)
    scatter!(p, [NaN], [NaN], label="pₙ(k)∝1/k²", markershape=:rect, color="lightgrey", msa = 1.0, msc="black", msw=1.5)
    scatter!(p, [NaN], [NaN], label="pₙ(k)∝1/k!", markershape=:diamond, color="lightgrey", msa = 1.0, msc="black", msw=1.5)
    #cols = [RGB(0.56, 0.94, 0.56), RGB(1.0, 0.52, 0.53), RGB(0.56, 0.83, 1.0)]
    cols = ["green", "red", "blue"]
    scatter!(p, [NaN], [NaN], label=" ", ms=0, mc=:white, msc=:white)
    scatter!(p, [NaN], [NaN], label="n = 100", color=cols[1], lwd=3.0, shape=:circle)
    scatter!(p, [NaN], [NaN], label="n = 1000", color=cols[2], lwd=3.0, shape=:circle)
    scatter!(p, [NaN], [NaN], label="n = 10000", color=cols[3], lwd=3.0, shape=:circle)
    mshapes = [:dtriangle, :rect, :diamond]
    for i in eachindex(ns)
        for j in eachindex(dists)
            for l = 2:4
                scatter!(p, [j+0.15*(i-2)], [log(risks[i,j,l]/risks[i,j,1])], markercolor=cols[i], label="", markershape=mshapes[l-1], msw=1.5, msc="black")
            end
        end
    end
    p
    savefig(p, joinpath(@__DIR__, "figures", "investigate_prior_k.pdf"))
end

function plot_risks_a()
    as = [0.01, 0.1, 1.0, 10.0, 100.0]
    ns = [100, 1000, 10000]
    dists = [Gamma(3.0, 3.0), Beta(3.0, 3.0), TDist(3.0)]
    risks = Array{Float64}(undef, length(ns), length(dists), 5)
    k_means = Array{Float64}(undef, length(ns), length(dists), 5)

    for i in eachindex(ns)
        for j in eachindex(dists)
            risks[i,j,:], k_means[i,j,:] = compute_risk_a(ns[i], dists[j], Xoshiro(1812))
        end
    end
    
    p1 = plot(xticks=(1:3, ["Gamma(3,3)", "Beta(3, 3)", "t(3)"]),
            ylabel="log relative risk", legend=:outerright, fg_legend=:transparent)
    p2 = plot(xticks=(1:5, ["n = 100", "n = 1000", "n = 10000"]),
    ylabel="mean number of bins relative to a = 1", legend=:outerright)
    p3 = plot(xaxis=:log10, xlabel="a", ylabel="Mean number of bins", legend=:topleft)
    #cols = [RGB(0.56, 0.94, 0.56), RGB(1.0, 0.52, 0.53), RGB(0.56, 0.83, 1.0)]
    cols = ["green", "red", "blue"]
    mshapes = [:dtriangle, :rect, nothing, :diamond, :utriangle]
    scatter!(p1, [NaN], [NaN], label="a = 0.01", markershape=mshapes[1], color="lightgrey", msa = 1.0, msc="black", msw=1.5)
    scatter!(p1, [NaN], [NaN], label="a = 0.1", markershape=mshapes[2], color="lightgrey", msa = 1.0, msc="black", msw=1.5)
    scatter!(p1, [NaN], [NaN], label="a = 10.0", markershape=mshapes[4], color="lightgrey", msa = 1.0, msc="black", msw=1.5)
    scatter!(p1, [NaN], [NaN], label="a = 100.0", markershape=mshapes[5], color="lightgrey", msa = 1.0, msc="black", msw=1.5)
    scatter!(p2, [NaN], [NaN], label="a = 0.01", markershape=mshapes[1], color="lightgrey", msa = 1.0, msc="black", msw=1.5)
    scatter!(p2, [NaN], [NaN], label="a = 0.1", markershape=mshapes[2], color="lightgrey", msa = 1.0, msc="black", msw=1.5)
    scatter!(p2, [NaN], [NaN], label="a = 10.0", markershape=mshapes[4], color="lightgrey", msa = 1.0, msc="black", msw=1.5)
    scatter!(p2, [NaN], [NaN], label="a = 100.0", markershape=mshapes[5], color="lightgrey", msa = 1.0, msc="black", msw=1.5)
    scatter!(p1, [NaN], [NaN], label=" ", ms=0, mc=:white, msc=:white)
    scatter!(p1, [NaN], [NaN], label="n = 100", color=cols[1], shape=:circle)
    scatter!(p1, [NaN], [NaN], label="n = 1000", color=cols[2], shape=:circle)
    scatter!(p1, [NaN], [NaN], label="n = 10000", color=cols[3], shape=:circle)
    for i in eachindex(ns)
        for j in eachindex(dists)
            for l = [1, 2, 4, 5]
                scatter!(p1, [j+0.15*(i-2)], [log(risks[i,j,l]/risks[i,j,3])], markercolor=cols[i], label="", markershape=mshapes[l], msw=1.5, msc="black")
            end
        end
        for l = [1, 2, 4, 5]
            scatter!(p2, [i], [mean(k_means[i,:,l])/mean(k_means[i,:,3])], markercolor=cols[i], label="", markershape=mshapes[l], msw=1.5, msc="black")
        end
        plot!(p3, as, mean(k_means[i,:,:], dims=1)', label="n = $(ns[i])", color=cols[i])
    end
    ylims!(p3, 2, 20)
    savefig(p1, joinpath(@__DIR__, "figures", "investigate_prior_a_risk.pdf"))
    savefig(p3, joinpath(@__DIR__, "figures", "investigate_prior_a_k_mean.pdf"))
end

plot_risks_k()
plot_risks_a()


