using AutoHist, Plots, StatsBase

function run_example()
    p_values = parse.(Float64, readlines(joinpath(@__DIR__, "hedenfalk.txt")))

    h1 = histogram_irregular(p_values; grid=:data, support=(0.0, 1.0), a = 5.0, alg=DP())
    p1 = plot(h1, alpha=0.5, xlabel="p", ylabel="Density", xlims=[-0.02, 1.02], label="", color="grey", title="Irregular histogram")

    savefig(p1, joinpath(@__DIR__, "figures", "hedenfalk_irregular.pdf"))

    # Estimate density at p = 1
    println("Estimated π₀, irregular: $(h1.density[end])")

    # Count number of observations in leftmost bin
    println("Number of observations in leftmost bin of irregular histogram $(h1.counts[1])")


    h2 = histogram_regular(p_values; a = k->0.5*k, support=(0.0, 1.0))
    p2 = plot(h2, alpha=0.5, xlabel="p", ylabel="Density", xlims=[-0.02, 1.02], label="", color="grey", title="Regular histogram")
    savefig(p2, joinpath(@__DIR__, "figures", "hedenfalk_regular.pdf"))

    println("Estimated π₀, regular: $(h2.density[end])")
end

run_example()