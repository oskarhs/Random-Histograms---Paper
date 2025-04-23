using Random, AutoHist, DataFrames, CSV

include("loss_functions.jl")
include(joinpath("R_methods", "wand_hist.jl"))
include(joinpath("R_methods", "RMG_hist.jl"))
include(joinpath("R_methods", "taut_string.jl"))

function get_methods()
    return [
        "Wand", "AIC", "BIC", "BR", "Knuth", "SC",
        "RIH", "RMG-B", "RMG-R", "TS", "L2CV", "KLCV"
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

function compute_losses(d::ContinuousUnivariateDistribution, breaks::AbstractVector{<:Real}, dens::AbstractVector{<:Real}; l2::Bool=false)
    loss_hell = hellinger_loss(d, breaks, dens)
    loss_pid, _, _, _ = peak_id_loss(d, breaks, dens)
    if l2
        loss_l2 = l2_loss(d, breaks, dens)
    else 
        loss_l2 = 0.0
    end
    return loss_hell, loss_pid, loss_l2
end

# Estimate risk for all methods for a sample of size n from a density f₀
function estimate_risk(rng, d, n, B; l2=false)
    methods = get_methods()
    num_methods = length(methods)    

    risk_hell = Array{Float64}(undef, num_methods)
    risk_l2 = Array{Float64}(undef, num_methods)
    risk_pid = Array{Float64}(undef, num_methods)
    loss_hell = Array{Float64}(undef, B, num_methods)
    loss_l2 = Array{Float64}(undef, B, num_methods)
    loss_pid = Array{Float64}(undef, B, num_methods)

    for b = 1:B
        x = rand(rng, d, n)
        breaks, dens = wand_hist(x)
        loss_hell[b,1], loss_pid[b,1], loss_l2[b,1] = compute_losses(d, breaks, dens; l2=l2)
        for j = 2:4
            # aic, bic and br
            H, _ = histogram_regular(x; rule=methods[j])
            breaks = collect(H.edges[1])
            loss_hell[b,j], loss_pid[b,j], loss_l2[b,j] = compute_losses(d, breaks, H.weights; l2=l2)
        end
        # Knuth
        H, _ = histogram_regular(x; rule="bayes", a = k->0.5*k)
        breaks = collect(H.edges[1])
        loss_hell[b,5], loss_pid[b,5], loss_l2[b,5] = compute_losses(d, breaks, H.weights; l2=l2)
        # SC
        H, _ = histogram_regular(x; rule="bayes", a = k->1.0*k)
        breaks = collect(H.edges[1])
        loss_hell[b,6], loss_pid[b,6], loss_l2[b,6] = compute_losses(d, breaks, H.weights; l2=l2)
        # RIH
        H, _ = histogram_irregular(x; rule="bayes", grid="regular")
        loss_hell[b,7], loss_pid[b,7], loss_l2[b,7] = compute_losses(d, H.edges[1], H.weights; l2=l2)
        # RMG-B
        breaks, dens = rmg_hist(x, "penB")
        loss_hell[b,8], loss_pid[b,8], loss_l2[b,8] = compute_losses(d, breaks, dens; l2=l2)
        # RMG-R
        breaks, dens = rmg_hist(x, "penR")
        loss_hell[b,9], loss_pid[b,9], loss_l2[b,9] = compute_losses(d, breaks, dens; l2=l2)
        # Taut String
        breaks, dens = taut_string(x; sorted=false)
        loss_hell[b,10], loss_pid[b,10], loss_l2[b,10] = compute_losses(d, breaks, dens; l2=l2)
        # L2CV
        H, _ = histogram_irregular(x; rule="l2cv", grid="data", use_min_length=true)
        loss_hell[b,11], loss_pid[b,11], loss_l2[b,11] = compute_losses(d, H.edges[1], H.weights; l2=l2)
        # KLCV
        H, _ = histogram_irregular(x; rule="klcv", grid="data", use_min_length=true)
        loss_hell[b,12], loss_pid[b,12], loss_l2[b,12] = compute_losses(d, H.edges[1], H.weights; l2=l2)
    end
    for j=1:num_methods
        risk_hell[j] = mean(loss_hell[:,j])
        risk_pid[j] = mean(loss_pid[:,j])
        risk_l2[j] = mean(loss_l2[:,j])
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

    ns = Int64[50, 200, 1000, 5000, 25000]
    B = 500

    colnms = vcat(["Density"], ["n"], methods)
    df_hell = DataFrame([[] for _ = colnms] , colnms)
    df_pid = DataFrame([[] for _ = colnms] , colnms)
    df_l2 = DataFrame([[] for _ = colnms] , colnms)

    for j in eachindex(test_densities)
        d = test_densities[j]
        for i in eachindex(ns)
            println("j = $j of $(length(test_densities)), i = $(i) of $(length(ns))")
            risk_hell, risk_pid, risk_l2 = estimate_risk(rng, d, ns[i], B; l2=in_l2[j])
            push!(df_hell, vcat(j, ns[i], risk_hell))
            push!(df_pid, vcat(j, ns[i], risk_pid))
            push!(df_l2, vcat(j, ns[i], risk_l2))
            #push!(df_hell, vcat(ifelse(i==1, "$j", ""), ns[i], risk_hell))
            #push!(df_pid, vcat(ifelse(i==1, "$j", ""), ns[i], risk_pid))
            #push!(df_l2, vcat(ifelse(i==1, "$j", ""), ns[i], risk_l2))
        end
    end
    # Write the data to csv files
    CSV.write(joinpath("simulations_data", "hellinger_risks.csv"), df_hell)
    CSV.write(joinpath("simulations_data", "pid_risks.csv"), df_pid)
    CSV.write(joinpath("simulations_data", "l2_risks.csv"), df_l2)
end

@time estimate_all_risks()

#@time estimate_risk(Xoshiro(1812), Beta(0.5, 0.5), 50, 500; l2=false)