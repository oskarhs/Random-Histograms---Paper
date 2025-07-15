using AutoHist, StatsBase, Distributions, MergeSorted, Plots

function hellinger_loss_hists(H1::AbstractHistogram, H2::AbstractHistogram)
    breaks1 = H1.edges[1]
    dens1 = H1.weights
    breaks2 = H2.edges[1]
    dens2 = H2.weights
    disc = mergesorted(breaks1, breaks2) # union of points of discontinuity for the two histograms
    m = length(disc)-1
    hell = 0.0
    for j = 1:m
        bin_ind1 = searchsortedfirst(breaks1, disc[j]+10*eps()) - 1
        bin_ind2 = searchsortedfirst(breaks2, disc[j]+10*eps()) - 1
        if bin_ind1 == 0 || bin_ind1 == (length(dens1)+1)
            h1 = 0.0
        else
            h1 = dens1[bin_ind1]
        end
        if bin_ind2 == 0 || bin_ind2 == (length(dens2)+1)
            h2 = 0.0
        else
            h2 = dens2[bin_ind2]
        end
        hell += (disc[j+1]-disc[j])*(sqrt(h1)-sqrt(h2))^2
    end
    return sqrt(hell)
end

function test_greedy()
    rng = Xoshiro(1812)
    avg_loss = 0.0
    B = 500
    for b = 1:B
        x = randn(rng, 3000)
        H1 = histogram_irregular(x; alg=GPDP(greedy=true))
        H2 = histogram_irregular(x; alg=DP(greedy=false))
        avg_loss += hellinger_loss_hists(H1, H2) / B
    end
    println("Average Hellinger loss: ", avg_loss)

    x = randn(rng, 3000)
    H1 = histogram_irregular(x; alg=GPDP(greedy=true))
    H2 = histogram_irregular(x; alg=DP(greedy=false))
    p = plot(H1, fillalpha=0.2, color="blue")
    plot!(p, H2, fillalpha=0.2, color="red")
end

test_greedy()