using Random, AutoHist, DataFrames, CSV

include(joinpath(@__DIR__, "loss_functions.jl"))
include(joinpath(@__DIR__, "R_methods", "hpi.jl"))
include(joinpath(@__DIR__, "R_methods", "modetest.jl"))
include(joinpath(@__DIR__, "R_methods", "taut_string.jl"))

# Run this from the simulation_study folder
# To speed up the code a little, open Julia with multiple threads.

function get_methods()
    return [
        "RIH", "SI", "FM", "WJ"
    ]
end

# Compute bins on a regular grid
function bin_regular(x::AbstractVector{<:Real}, xmin::Real, xmax::Real, k::Int, right::Bool)
    R = xmax - xmin
    bincounts = zeros(Float64, k)
    edges_inc = k/R
    if right
        for val in x
            idval = min(k-1, floor(Int, (val-xmin)*edges_inc+eps())) + 1
            @inbounds bincounts[idval] += 1.0
        end
    else
        for val in x
            idval = max(0, floor(Int, (val-xmin)*edges_inc-eps())) + 1
            @inbounds bincounts[idval] += 1.0
        end
    end
    return bincounts
end

# For histograms
function compute_losses(d::ContinuousUnivariateDistribution, breaks::AbstractVector{<:Real}, dens::AbstractVector{<:Real})
    loss_pid, num_est_modes = peak_id_loss(d, breaks, dens)
    return loss_pid, num_est_modes
end

# For kernel estimators
function compute_losses(d::ContinuousUnivariateDistribution, est_modes::AbstractVector{<:Real})
    loss_pid, num_est_modes = peak_id_loss(d, est_modes)
    return loss_pid, num_est_modes
end

# Estimate risk for all methods for a sample of size n from a density f₀
function estimate_risk(rng, d, n, B)
    methods = get_methods()
    num_methods = length(methods)    

    mean_num_modes = Vector{Float64}(undef, num_methods)
    risk_pid = Array{Float64}(undef, num_methods)
    loss_hell = Array{Float64}(undef, B, num_methods)
    num_modes = Array{Float64}(undef, B, num_methods)
    elapsed = Matrix{Float64}(undef, B, num_methods)
    median_elapsed = Vector{Float64}(undef, num_methods)

    for b = 1:B
        x = rand(rng, d, n)
        # RIH
        elapsed[b,1] = @elapsed(H = histogram_irregular(x; rule=:bayes, grid=:data, a = 5.0, alg = DP()))
        loss_pid[b,1], num_modes[b,2] = compute_losses(d, H.breaks, H.density)
        # SI
        est_modes, elapsed[b,2] = modetest(x; method="SI")
        loss_pid[b,2], num_modes[b,2] = compute_losses(d, est_modes)
        # FM
        est_modes, elapsed[b,3] = modetest(x; method="FM")
        loss_pid[b,3], num_modes[b,3] = compute_losses(d, est_modes)
        # WJ
        est_modes, elapsed[b,4] = hpi(x)
        loss_pid[b,4], num_modes[b,4] = compute_losses(d, est_modes)
    end
    for j=1:num_methods
        risk_pid[j] = mean(loss_pid[:,j])
        median_elapsed[j] = median(elapsed[:,j])
        mean_num_modes[j] = mean(num_modes[:,j])
    end
    return risk_hell, risk_pid, risk_l2
end

function estimate_all_risks()
    methods = get_methods()
    test_densities = get_test_distributions()
    in_l2 = Bool[
        true, true, false, true, true, false, true, true,
        true, true, true, true, true, true, true, true,
    ]
    rng = Xoshiro(1812)

    ns = Int64[200, 1000, 5000]
    B = 200

    colnms = vcat(["Density"], ["n"], methods)
    df_hell = DataFrame([[] for _ = colnms] , colnms)
    df_pid = DataFrame([[] for _ = colnms] , colnms)
    df_l2 = DataFrame([[] for _ = colnms] , colnms)

    for j in eachindex(test_densities)
        d = test_densities[j]
        for i in eachindex(ns)
            println("j = $j of $(length(test_densities)), i = $(i) of $(length(ns))")
            risk_hell, risk_pid, risk_l2 = estimate_risk(rng, d, ns[i], B)
            push!(df_hell, vcat(j, ns[i], risk_hell))
            push!(df_pid, vcat(j, ns[i], risk_pid))
            push!(df_l2, vcat(j, ns[i], risk_l2))
        end
    end
    # Write the data to csv files
    CSV.write(joinpath(@__DIR__, "simulations_data", "hellinger_risks.csv"), df_hell)
    CSV.write(joinpath(@__DIR__, "simulations_data", "pid_risks.csv"), df_pid)
    CSV.write(joinpath(@__DIR__, "simulations_data", "l2_risks.csv"), df_l2)
end

@time estimate_all_risks()

#r = estimate_risk(Xoshiro(1812), Chisq(1.0), 25000, 500; l2=false)
#println(r[1])

#println(isfile(joinpath(@__DIR__, "simulations_data", "hellinger_risks.csv")))