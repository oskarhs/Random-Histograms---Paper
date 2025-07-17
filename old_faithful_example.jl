using AutoHist, Plots, StatsBase, RCall

function run_faithful_example()
    waiting = parse.(Float64, readlines(joinpath(@__DIR__, "old_faithful.txt")))

    # Compute the binned KDE of Wand and Jones
    kernel_est = convert(
        Array{Float64}, 
        R"""
        dens = density($waiting, bw="SJ")
        matrix(c(dens$x, dens$y), ncol=2)
        """
    )

    h1 = histogram_irregular(waiting; grid=:data, a=5.0, alg=DP())
    p1 = plot(h1, alpha=0.3, xlabel="Waiting time", ylabel="Density", label="", color="grey", title="Irregular histogram")
    plot!(p1, kernel_est[:,1], kernel_est[:,2], color="blue", linestyle=:dash, label="", lw=2.0)
    ylims!(p1, 0.0, 0.045)
    xlims!(p1, 38.0, 110.0)


    println("Number of observations in second bin of irregular histogram $(h1.counts[2])")
    println("Number of observations in each bin of irregular $(h1.counts)")


    h2 = histogram_regular(waiting; a=k->0.5*k)
    p2 = plot(h2, alpha=0.3, xlabel="Waiting time", ylabel="Density", label="", color="grey", title="Regular histogram")
    plot!(p2, kernel_est[:,1], kernel_est[:,2], color="blue", linestyle=:dash, label="", lw=2.0)
    ylims!(p2, 0.0, 0.045)
    xlims!(p2, 38.0, 110.0)

    savefig(p1, joinpath(@__DIR__, "figures", "old_faithful_irregular.pdf"))
    savefig(p2, joinpath(@__DIR__, "figures", "old_faithful_regular.pdf"))
end

run_faithful_example()