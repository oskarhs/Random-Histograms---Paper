using AutoHist, Plots

include(joinpath("simulation_study", "loss_functions.jl"))
include(joinpath("simulation_study", "test_distributions.jl"))

function generate_harp_mode_plot()
    rng = Xoshiro(6)
    d = Harp()
    n = 5000
    x = rand(rng, d, n)

    d_modes = modes(Harp())
    delta = pid_tolerance(Harp())

    H_reg = histogram_regular(x; rule="bic")
    loss_reg, reg_modes, reg_vals, spur_reg = peak_id_loss(d, H_reg.edges[1], H_reg.weights)
    println("PID loss, regular: $loss_reg")
    println("Number of spurious modes, regular: $spur_reg")

    println("Hellinger loss, regular: ", round(hellinger_loss(d, collect(H_reg.edges[1]), H_reg.weights); sigdigits=3))

    H_irreg = histogram_irregular(x; rule="bayes", a=5.0)
    loss_irreg, irreg_modes, irreg_vals, spur_irreg = peak_id_loss(d, H_irreg.edges[1], H_irreg.weights)
    println("PID loss, irregular: $loss_irreg")
    println("Number of spurious modes, irregular: $spur_irreg")

    println("Hellinger loss, irregular: ", round(hellinger_loss(d, H_irreg.edges[1], H_irreg.weights); digits=3))

    # Generate plots
    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

    t = LinRange(-3, 90, 10000)
    p1 = plot(xlabel="x", ylabel="Density", title="BIC")
    plot!(p1, t, pdf.(d, t), lw=2.0, color="blue", label="")
    plot!(p1, H_reg, fillalpha=0.25, normalize=:pdf, color="red", label="", linecolor="red")
    xlims!(p1, -3.5, 90.5)
    ylims!(p1, -0.005, 0.17)
    for j in eachindex(d_modes)
        plot!(p1, rectangle(2.0*delta[j], 0.18, d_modes[j]-delta[j], 0.0), opacity=0.15, color="black", label="")
    end
    scatter!(p1, reg_modes[[1,2,3,5,10]], zeros(4), color="red", label="", linecolor="red")
    scatter!(p1, reg_modes[setdiff(1:end, [1,2,3,5,10])], zeros(10), color="black", label="")

    p2 = plot(xlabel="x", ylabel="Density", title="Random irregular histogram")
    plot!(p2, t, pdf.(d, t), lw=2.0, color="blue", label="")
    plot!(p2, H_irreg, fillalpha=0.25, normalize=:pdf, color="red", label="", linecolor="red")
    xlims!(p2, -3.5, 90.5)
    ylims!(p2, -0.005, 0.17)
    for j in eachindex(d_modes)
        plot!(p2, rectangle(2.0*delta[j], 0.18, d_modes[j]-delta[j], 0.0), opacity=0.15, color="black", label="")
    end
    scatter!(p2, irreg_modes, zeros(length(irreg_modes)), color="red", label="", linecolor="red")
    p3 = plot(p1, p2, layout=(2,1), margin = 0.5*Plots.cm, size=(900, 800))
    savefig(p3, joinpath("figures", "HarpModeExample.pdf"))
end

p1, p2 = generate_harp_mode_plot()