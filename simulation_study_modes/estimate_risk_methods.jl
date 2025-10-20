using Random, AutoHist, DataFrames, CSV, ProgressMeter, Statistics

include(joinpath(@__DIR__, "loss_functions.jl"))
include(joinpath(@__DIR__, "R_methods", "hpi.jl"))
include(joinpath(@__DIR__, "R_methods", "multimode.jl"))
include(joinpath(@__DIR__, "R_methods", "taut_string.jl"))

# Run this from the simulation_study folder
# To speed up the code a little, open Julia with multiple threads.

function get_methods()
    return [
        "RIH", "SI", "FM", "WJ"
    ]
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
function estimate_risk(rng, d, n, B; progress=true)
    methods = get_methods()
    num_methods = length(methods)    

    mean_num_modes = Vector{Float64}(undef, num_methods)
    risk_pid = Array{Float64}(undef, num_methods)
    median_elapsed = Vector{Float64}(undef, num_methods)
    sd_pid = Vector{Float64}(undef, num_methods)
    sd_elapsed = Vector{Float64}(undef, num_methods)
    loss_pid = Array{Float64}(undef, B, num_methods)
    num_modes = Array{Float64}(undef, B, num_methods)
    elapsed = Matrix{Float64}(undef, B, num_methods)

    failed = fill(false, length(methods))

    # Progressbar
    pm = progress ? Progress(B; desc="Computing PID risk:", barlen=31) : nothing

    for b = 1:B
        #println(b)
        x = rand(rng, d, n)
        # RIH
        elapsed[b,1] = @elapsed(H = histogram_irregular(x; rule=:bayes, grid=:data, a = 5.0, alg = DP()))
        loss_pid[b,1], num_modes[b,1] = compute_losses(d, H.breaks, H.density)
        # SI
        est_modes, elapsed[b,2] = modetest(x; method="SI")
        if any(ismissing.(est_modes))
            failed[2] = true
        else
            loss_pid[b,2], num_modes[b,2] = compute_losses(d, est_modes)
        end
        # FM
        est_modes, elapsed[b,3] = modetest(x; method="FM")
        if any(ismissing.(est_modes))
            failed[3] = true
        else
            loss_pid[b,3], num_modes[b,3] = compute_losses(d, est_modes)
        end

        # WJ
        est_modes, elapsed[b,4] = hpi(x)
        if any(ismissing.(est_modes))
            failed[4] = true
        else
            loss_pid[b,4], num_modes[b,4] = compute_losses(d, est_modes)
        end

        if !isnothing(pm)
            next!(pm; showvalues=[("Iteration", b, "of", B)])
        end
    end
    for j=1:num_methods
        risk_pid[j] = mean(loss_pid[:,j])
        median_elapsed[j] = median(elapsed[:,j])
        mean_num_modes[j] = mean(num_modes[:,j])
        sd_pid[j] = std(loss_pid[:,j])
        sd_elapsed[j] = std(elapsed[:,j])
    end
    return risk_pid, median_elapsed, mean_num_modes, sd_pid, sd_elapsed, failed
end

function estimate_all_risks()
    methods = get_methods()
    test_densities = get_test_distributions()
    rng = Xoshiro(1812)

    println("Estimating risks for $test_densities. This will take a while...")

    ns = Int64[200, 600, 2000]
    B = 200

    colnms = vcat(["Density"], ["n"], methods)
    df_pid = DataFrame([[] for _ = colnms] , colnms)
    df_elapsed = DataFrame([[] for _ = colnms] , colnms)
    df_num_modes = DataFrame([[] for _ = colnms] , colnms)
    df_sd_pid = DataFrame([[] for _ = colnms] , colnms)
    df_sd_elapsed = DataFrame([[] for _ = colnms] , colnms)
    df_failed = DataFrame([[] for _ = colnms] , colnms)

    for j in eachindex(test_densities)
        d = test_densities[j]
        for i in eachindex(ns)
            println("j = $j of $(length(test_densities)), i = $(i) of $(length(ns))")
            risk_pid, median_elapsed, mean_num_modes, sd_pid, sd_elapsed, failed = estimate_risk(rng, d, ns[i], B)
            push!(df_pid, vcat(j, ns[i], risk_pid))
            push!(df_elapsed, vcat(j, ns[i], median_elapsed))
            push!(df_num_modes, vcat(j, ns[i], mean_num_modes))
            push!(df_sd_pid, vcat(j, ns[i], sd_pid))
            push!(df_sd_elapsed, vcat(j, ns[i], sd_elapsed))
            push!(df_failed, vcat(j, ns[i], failed))
        end
    end
    # Write the data to csv files
    try
        mkdir(joinpath(@__DIR__, "simulations_data"))
    catch
        nothing
    end
    CSV.write(joinpath(@__DIR__, "simulations_data", "pid_risks.csv"), df_pid)
    CSV.write(joinpath(@__DIR__, "simulations_data", "elapsed.csv"), df_elapsed)
    CSV.write(joinpath(@__DIR__, "simulations_data", "num_modes.csv"), df_num_modes)
    CSV.write(joinpath(@__DIR__, "simulations_data", "sd_pid.csv"), df_sd_pid)
    CSV.write(joinpath(@__DIR__, "simulations_data", "sd_elapsed.csv"), df_sd_elapsed)
    CSV.write(joinpath(@__DIR__, "simulations_data", "df_failed.csv"), df_failed)
end

#estimate_all_risks()

r = estimate_risk(Xoshiro(1812), LogNormal(), 200, 200)
println(r)
#println(r[1])

#println(isfile(joinpath(@__DIR__, "simulations_data", "hellinger_risks.csv")))