using AutoHist, Plots, DataFrames, CSV, Latexify, StatsPlots

function generate_risk_tables()
    methods = [
        "Wand", "AIC", "BIC", "BR", "Knuth", "SC",
        "RIH", "RMG-B", "RMG-R", "TS", "L2CV", "KLCV"
    ]
    ns = Int64[50, 200, 1000, 5000, 25000]
    df_hell = CSV.read(joinpath("simulations_data", "hellinger_risks.csv"), DataFrame)
    df_pid = CSV.read(joinpath("simulations_data", "pid_risks.csv"), DataFrame)
    df_l2 = CSV.read(joinpath("simulations_data", "l2_risks.csv"), DataFrame)

    # Convert some columns to integer
    df_hell[!, :n] = convert.(Int64, df_hell[!, :n])
    df_hell[!, :Density] = convert.(Int64, df_hell[!, :Density])
    df_pid[!, :n] = convert.(Int64, df_pid[!, :n])
    df_pid[!, :Density] = convert.(Int64, df_pid[!, :Density])
    df_l2[!, :n] = convert.(Int64, df_l2[!, :n])
    df_l2[!, :Density] = convert.(Int64, df_l2[!, :Density])

    # First, create all risk tables to be written to file
    colnms = vcat(["Density"], ["n"], methods)
    df_hell_print = DataFrame([[] for _ = colnms] , colnms)
    df_pid_print = DataFrame([[] for _ = colnms] , colnms)
    df_l2_print = DataFrame([[] for _ = colnms] , colnms)
    for j in 1:16
        for i in eachindex(ns)
            push!(df_hell_print, df_hell[5*(j-1)+i,:])
            push!(df_pid_print, df_pid[5*(j-1)+i,:])
            if j ∉ [3, 6] # these densities are not in l2
                push!(df_l2_print, df_l2[5*(j-1)+i,:])
            end
            if i != 1
                df_hell_print[end,1] = ""
                if j ∉ [3, 6] # these densities are not in l2
                    df_l2_print[end,1] = ""
                end
                df_pid_print[end,1] = ""
            end
            df_hell_print[end,3:end] = round.(values(df_hell_print[end,3:end]); digits=3)
            if j ∉ [3, 6] # these densities are not in l2
                df_l2_print[end,3:end] = round.(values(df_l2_print[end,3:end]); digits=3)
            end
            df_pid_print[end,3:end] = round.(values(df_pid_print[end,3:end]); digits=3)
        end
    end

    hell_table = latexify(df_hell_print; env = :table, booktabs = true, latex = false)
    hell_table_vec = split(hell_table, "\n")
    j = 4
    for i in 1:15
        hell_table_vec[j+5*i] = hell_table_vec[j+5*i] * " \\hline" 
    end
    hell_table = join(hell_table_vec, "\n")
    open(joinpath("simulations_data", "risk_tables", "hellinger_risk_table.txt"), "w") do io
        println(io, hell_table)
    end

    pid_table = latexify(df_pid_print; env = :table, booktabs = true, latex = false)
    pid_table_vec = split(pid_table, "\n")
    for i in 1:15
        pid_table_vec[j+5*i] = pid_table_vec[j+5*i] * " \\hline" 
    end
    pid_table = join(pid_table_vec, "\n")
    open(joinpath("simulations_data", "risk_tables", "pid_risk_table.txt"), "w") do io
        println(io, pid_table)
    end

    l2_table = latexify(df_l2_print; env = :table, booktabs = true, latex = false)
    l2_table_vec = split(l2_table, "\n")
    for i in 1:13 # two densities are not in L2
        l2_table_vec[j+5*i] = l2_table_vec[j+5*i] * " \\hline" 
    end
    l2_table = join(l2_table_vec, "\n")
    open(joinpath("simulations_data", "risk_tables", "l2_risk_table.txt"), "w") do io
        println(io, l2_table)
    end
end


function generate_lrr_figure(n)
    methods = [
        "Wand", "AIC", "BIC", "BR", "Knuth", "SC",
        "RIH", "RMG-B", "RMG-R", "TS", "L2CV", "KLCV"
    ]
    ns = Int64[50, 200, 1000, 5000, 25000]
    if n ∉ ns
        return nothing
    elseif n == 50
        ind = 1
    elseif n == 200
        ind = 2
    elseif n == 1000
        ind = 3
    elseif n == 5000
        ind = 4
    elseif n == 25000
        ind = 5
    end

    in_l2 = Bool[
        true, true, false, true, true, false, true, true,
        true, true, true, true, true, true, true, true,
    ]

    df_hell = CSV.read(joinpath("simulations_data", "hellinger_risks.csv"), DataFrame)
    df_l2 = CSV.read(joinpath("simulations_data", "l2_risks.csv"), DataFrame)

    # Convert some columns to integer
    df_hell[!, :n] = convert.(Int64, df_hell[!, :n])
    df_hell[!, :Density] = convert.(Int64, df_hell[!, :Density])
    df_l2[!, :n] = convert.(Int64, df_l2[!, :n])
    df_l2[!, :Density] = convert.(Int64, df_l2[!, :Density])

    # Lazy implementation
    lrr_hell = Array{Float64}(undef, 16, length(methods))
    lrr_l2 = Array{Float64}(undef, 14, length(methods)) # two densities not in l2

    j = 1
    for i in 1:16
        risks_hell = values(df_hell[ind + 5*(i-1),3:end])
        lrr_hell[i,:] .= log.(risks_hell ./ minimum(risks_hell))
        if in_l2[i]
            risks_l2 = values(df_l2[ind + 5*(i-1),3:end])
            lrr_l2[j,:] .= log.(risks_l2 ./ minimum(risks_l2))
            j = j+1
        end
    end
    df_lrr_hell = DataFrame(lrr_hell, methods)
    df_lrr_l2 = DataFrame(lrr_l2, methods)
    p1 = @df df_lrr_hell boxplot(cols(), legend=false, color="black", fillalpha=0.3)
    plot!(p1, xticks=(1:length(methods), methods), color="black", fillalpha=0.3, ylabel="LRRₙ(f₀, m)", title="Hellinger risk, n = $n")
    p2 = @df df_lrr_l2 boxplot(cols(), legend=false, color="black", fillalpha=0.3)
    plot!(p2, xticks=(1:length(methods), methods), color="black", fillalpha=0.3, ylabel="LRRₙ(f₀, m)", title="L2 risk, n = $n")

    savefig(p1, joinpath("simulations_data", "figures", "lrr_hell_$n.pdf"))
    savefig(p2, joinpath("simulations_data", "figures", "lrr_l2_$n.pdf"))
end

generate_risk_tables()

for n in Int64[50, 200, 1000, 5000, 25000]
    generate_lrr_figure(n)
end