using Random, Distributions, Latexify, AutoHist, DataFrames

function sample_p_vals(rng, π0, n, β)
    x = Vector{Float64}(undef, n)
    for i = 1:n
        u = rand(rng, Uniform(0.0, 1.0))
        if u < π0
            x[i] = rand(rng, Uniform(0.0, 1.0))
        else
            x[i] = rand(rng, Beta(1.0, β))
        end
    end
    return x
end

function estimate_risks(β)
    rng = Xoshiro(1812)

    ns = [200, 1000, 5000]
    π0s = [0.5, 0.8, 0.95]

    B = 500
    rmse_irr = Vector{Float64}(undef, B)
    rmse_reg = Vector{Float64}(undef, B)

    names = ["pi0", "n", "Average RMSE, irregular", "Average RMSE, regular"]
    risks = DataFrame([[] for _ = names] , names)
    for j in eachindex(π0s)
        for i in eachindex(ns)
            println("j = ", j, ", i = ", i)
            for b = 1:B
                x = sample_p_vals(rng, π0s[j], ns[i], β)
                H_irr = histogram_irregular(x; support=(0.0, 1.0), a=5.0, grid=:data, alg = ifelse(ns[i] ≥ 500, GPDP(), DP()))
                H_reg = histogram_regular(x; support=(0.0, 1.0), a=k->0.5*k)
                π0_irr = H_irr.density[end] # estimated density in last bin
                π0_reg = H_reg.density[end]
                rmse_irr[b] = sqrt(mean((π0s[j] - π0_irr)^2))
                rmse_reg[b] = sqrt(mean((π0s[j] - π0_reg)^2))
            end
            risks = vcat(
                risks, 
                DataFrame(
                    "pi0" => ifelse(i == 1, string(π0s[j]), ""),
                    "n" => ns[i],
                    "Average RMSE, irregular" => round(mean(rmse_irr); digits=3),
                    "Average RMSE, regular" => round(mean(rmse_reg); digits=3)
                )
            )
        end
    end
    latexify(risks; env = :table, booktabs = true, latex = false) |> println
end

#estimate_risks(5.0)
estimate_risks(4.0)
#estimate_risks(10.0)