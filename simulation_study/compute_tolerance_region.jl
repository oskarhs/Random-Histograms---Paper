using Plots, Integrals, FastGaussQuadrature

function l1_loss_segment(dens_hist, bin_edges, dist)
    ip = IntegralProblem((t,p) -> abs(pdf(dist, t) - dens_hist), bin_edges[1], bin_edges[2])
    iae = solve(ip, GaussLegendre()).u
    return iae
end

function const_approx_mode(mode_x, d, δ)
    dens_hist = 1.0/(2.0*δ)*(cdf(d, mode_x+δ) - cdf(d, mode_x-δ))
    local_peakedness = l1_loss_segment(dens_hist, [mode_x-δ, mode_x+δ], d) / (cdf(d, mode_x+δ) - cdf(d, mode_x-δ))
    return local_peakedness
end

function test_peakness_measure(d)
    δ = Array{Float64}(undef, length(modes(d)))
    s = 0.5
    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    #t = LinRange(-3.5, 3.5, 10000)
    t = LinRange(-2.0, 80, 10000)
    p = plot(t, pdf.(d, t))
    modes_d = modes(d)
    for j in eachindex(modes_d)
        δ[j] = 0.0001
        locpeak = const_approx_mode(modes_d[j], d, δ[j])
        while locpeak < 0.2
            δ[j] += 1e-4
            locpeak = const_approx_mode(modes_d[j], d, δ[j])
        end
        println("j = $j, locpeak: $(locpeak)")
        plot!(p, [modes_d[j] - s*δ[j], modes_d[j] + s*δ[j]], [pdf(d, modes_d[j]), pdf(d, modes_d[j])], label="")
        plot!(p, rectangle(2.0*δ[j]*s, pdf(d, modes_d[j]), modes_d[j]-δ[j]*s, 0.0), opacity=0.15, color="black", label="")
        #plot!(p, rectangle(2.0*δ[j]*s, 0.16, modes_d[j]-δ[j]*s, 0.0), opacity=0.15, color="black", label="")

    end
    display(p)
    return 0.5*δ, p
end
|
δ, p = test_peakness_measure(Harp())
@show δ
#test_peakness_measure(Harp())