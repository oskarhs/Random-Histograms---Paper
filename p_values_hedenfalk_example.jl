using AutoHist, Plots, StatsBase

function run_example()
    p_values = parse.(Float64, readlines("hedenfalk.txt"))

    H1, _ = histogram_irregular(p_values; grid="quantile", support=(0.0, 1.0))
    p1 = plot(H1, alpha=0.5, xlabel="p", ylabel="Density", xlims=[-0.02, 1.02], label="", color="grey", title="Irregular histogram")

    savefig(p1, joinpath("figures", "hedenfalk_irregular.pdf"))

    # Estimate density at p = 1
    println("Estimated π₀, irregular: $(H1.weights[end])")

    # Count number of observations in leftmost bin
    println("Number of observations in leftmost bin if irregular histogram $(fit(Histogram, p_values, H1.edges[1]).weights[1])")


    H2, _ = histogram_regular(p_values; a = k->0.5*k, support=(0.0, 1.0))
    p2 = plot(H2, alpha=0.5, xlabel="p", ylabel="Density", xlims=[-0.02, 1.02], label="", color="grey", title="Regular histogram")
    savefig(p2, joinpath("figures", "hedenfalk_regular.pdf"))

    println("Estimated π₀, regular: $(H2.weights[end])")
end

run_example()