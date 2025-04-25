using AutoHist, Plots, StatsBase, RCall

function run_faithful_example()
    waiting = parse.(Float64, readlines("old_faithful.txt"))

    # Compute the binned KDE of Wand and Jones
    binned_est = convert(
        Array{Float64}, 
        R"""
        dens = density($waiting, bw="SJ")
        matrix(c(dens$x, dens$y), ncol=2)
        """
    )
    #        dens = KernSmooth::bkde($waiting, kernel="epanech")


    H1, _ = histogram_irregular(waiting; grid="regular")
    p1 = plot(H1, alpha=0.3, xlabel="Waiting time", ylabel="Density", label="", color="grey", title="Irregular histogram")
    plot!(p1, binned_est[:,1], binned_est[:,2], color="blue", linestyle=:dash, label="", lw=2.0)
    ylims!(p1, 0.0, 0.045)
    xlims!(p1, 38.0, 110.0)


    println("Number of observations in second bin of irregular histogram $(fit(Histogram, waiting, H1.edges[1]).weights[2])")
    println("Number of observations in each bin of irregular $(fit(Histogram, waiting, H1.edges[1]).weights)")


    H2, _ = histogram_regular(waiting; a=k->0.5*k)
    p2 = plot(H2, alpha=0.3, xlabel="Waiting time", ylabel="Density", label="", color="grey", title="Regular histogram")
    plot!(p2, binned_est[:,1], binned_est[:,2], color="blue", linestyle=:dash, label="", lw=2.0)
    ylims!(p2, 0.0, 0.045)
    xlims!(p2, 38.0, 110.0)

    savefig(p1, joinpath("figures", "old_faithful_irregular.pdf"))
    savefig(p2, joinpath("figures", "old_faithful_regular.pdf"))
end

run_faithful_example()