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
    df_hell_print_1 = DataFrame([[] for _ = colnms] , colnms)
    df_hell_print_2 = DataFrame([[] for _ = colnms] , colnms)

    df_pid_print_1 = DataFrame([[] for _ = colnms] , colnms)
    df_pid_print_2 = DataFrame([[] for _ = colnms] , colnms)

    df_l2_print_1 = DataFrame([[] for _ = colnms] , colnms)
    df_l2_print_2 = DataFrame([[] for _ = colnms] , colnms)

    for j in 1:8
        for i in eachindex(ns)
            push!(df_hell_print_1, df_hell[5*(j-1)+i,:])
            push!(df_pid_print_1, df_pid[5*(j-1)+i,:])
            if j ∉ [3, 6] # these densities are not in l2
                push!(df_l2_print_1, df_l2[5*(j-1)+i,:])
            end
            if i != 1
                df_hell_print_1[end,1] = ""
                if j ∉ [3, 6] # these densities are not in l2
                    df_l2_print_1[end,1] = ""
                end
                df_pid_print_1[end,1] = ""
            end
            df_hell_print_1[end,3:end] = round.(values(df_hell_print_1[end,3:end]); digits=3)
            if j ∉ [3, 6] # these densities are not in l2
                df_l2_print_1[end,3:end] = round.(values(df_l2_print_1[end,3:end]); digits=3)
            end
            df_pid_print_1[end,3:end] = round.(values(df_pid_print_1[end,3:end]); digits=3)
        end
    end

    for j in 9:16
        for i in eachindex(ns)
            push!(df_hell_print_2, df_hell[5*(j-1)+i,:])
            push!(df_pid_print_2, df_pid[5*(j-1)+i,:])
            if j ∉ [3, 6] # these densities are not in l2
                push!(df_l2_print_2, df_l2[5*(j-1)+i,:])
            end
            if i != 1
                df_hell_print_2[end,1] = ""
                if j ∉ [3, 6] # these densities are not in l2
                    df_l2_print_2[end,1] = ""
                end
                df_pid_print_2[end,1] = ""
            end
            df_hell_print_2[end,3:end] = round.(values(df_hell_print_2[end,3:end]); digits=3)
            if j ∉ [3, 6] # these densities are not in l2
                df_l2_print_2[end,3:end] = round.(values(df_l2_print_2[end,3:end]); digits=3)
            end
            df_pid_print_2[end,3:end] = round.(values(df_pid_print_2[end,3:end]); digits=3)
        end
    end

    # Hellinger table part 1
    hell_table = latexify(df_hell_print_1; env = :table, booktabs = true, latex = false)
    hell_table_vec = split(hell_table, "\n")
    j = 4
    for i in 1:7
        hell_table_vec[j+5*i] = hell_table_vec[j+5*i] * " \\hline" 
    end
    hell_table = join(hell_table_vec, "\n")
    open(joinpath("simulations_data", "risk_tables", "hellinger_risk_table_1.txt"), "w") do io
        println(io, hell_table)
    end
    # Hellinger table part 2
    hell_table = latexify(df_hell_print_2; env = :table, booktabs = true, latex = false)
    hell_table_vec = split(hell_table, "\n")
    j = 4
    for i in 1:7
        hell_table_vec[j+5*i] = hell_table_vec[j+5*i] * " \\hline" 
    end
    hell_table = join(hell_table_vec, "\n")
    open(joinpath("simulations_data", "risk_tables", "hellinger_risk_table_2.txt"), "w") do io
        println(io, hell_table)
    end

    # Pid table part 1
    pid_table = latexify(df_pid_print_1; env = :table, booktabs = true, latex = false)
    pid_table_vec = split(pid_table, "\n")
    for i in 1:7
        pid_table_vec[j+5*i] = pid_table_vec[j+5*i] * " \\hline" 
    end
    pid_table = join(pid_table_vec, "\n")
    open(joinpath("simulations_data", "risk_tables", "pid_risk_table_1.txt"), "w") do io
        println(io, pid_table)
    end
    # Pid table part 2
    pid_table = latexify(df_pid_print_2; env = :table, booktabs = true, latex = false)
    pid_table_vec = split(pid_table, "\n")
    for i in 1:7
        pid_table_vec[j+5*i] = pid_table_vec[j+5*i] * " \\hline" 
    end
    pid_table = join(pid_table_vec, "\n")
    open(joinpath("simulations_data", "risk_tables", "pid_risk_table_2.txt"), "w") do io
        println(io, pid_table)
    end

    # L2 table part 1
    l2_table = latexify(df_l2_print_1; env = :table, booktabs = true, latex = false)
    l2_table_vec = split(l2_table, "\n")
    for i in 1:5 # two densities are not in L2
        l2_table_vec[j+5*i] = l2_table_vec[j+5*i] * " \\hline" 
    end
    l2_table = join(l2_table_vec, "\n")
    open(joinpath("simulations_data", "risk_tables", "l2_risk_table_1.txt"), "w") do io
        println(io, l2_table)
    end
    # L2 table part 2
    l2_table = latexify(df_l2_print_2; env = :table, booktabs = true, latex = false)
    l2_table_vec = split(l2_table, "\n")
    for i in 1:7 # two densities are not in L2
        l2_table_vec[j+5*i] = l2_table_vec[j+5*i] * " \\hline" 
    end
    l2_table = join(l2_table_vec, "\n")
    open(joinpath("simulations_data", "risk_tables", "l2_risk_table_2.txt"), "w") do io
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
    df_pid = CSV.read(joinpath("simulations_data", "pid_risks.csv"), DataFrame)
    df_l2 = CSV.read(joinpath("simulations_data", "l2_risks.csv"), DataFrame)

    # Convert some columns to integer
    df_hell[!, :n] = convert.(Int64, df_hell[!, :n])
    df_hell[!, :Density] = convert.(Int64, df_hell[!, :Density])
    df_pid[!, :n] = convert.(Int64, df_pid[!, :n])
    df_pid[!, :Density] = convert.(Int64, df_pid[!, :Density])
    df_l2[!, :n] = convert.(Int64, df_l2[!, :n])
    df_l2[!, :Density] = convert.(Int64, df_l2[!, :Density])

    # Lazy implementation
    lrr_hell = Array{Float64}(undef, 16, length(methods))
    r_pid = Array{Float64}(undef, 16, 6)
    lrr_l2 = Array{Float64}(undef, 14, length(methods)) # two densities not in l2

    j = 1
    for i in 1:16
        risks_hell = values(df_hell[ind + 5*(i-1),3:end])
        lrr_hell[i,:] .= log.(risks_hell ./ minimum(risks_hell))
        r_pid[i,:] .=  values(df_pid[ind + 5*(i-1),9:end])
        if in_l2[i]
            risks_l2 = values(df_l2[ind + 5*(i-1),3:end])
            lrr_l2[j,:] .= log.(risks_l2 ./ minimum(risks_l2))
            j = j+1
        end
    end
    df_lrr_hell = DataFrame(lrr_hell, methods)
    df_r_pid = DataFrame(r_pid, methods[7:end])
    df_lrr_l2 = DataFrame(lrr_l2, methods)
    p1 = @df df_lrr_hell boxplot(cols(), legend=false, color="black", fillalpha=0.3)
    plot!(p1, xticks=(1:length(methods), methods), color="black", fillalpha=0.3, ylabel="LRRₙ(f₀, m)", title="Hellinger risk, n = $n")
    p2 = @df df_lrr_l2 boxplot(cols(), legend=false, color="black", fillalpha=0.3)
    plot!(p2, xticks=(1:length(methods), methods), color="black", fillalpha=0.3, ylabel="LRRₙ(f₀, m)", title="L2 risk, n = $n")
    p3 = @df df_r_pid[:,1:4] boxplot(cols(), legend=false, color="black", fillalpha=0.3)
    plot!(p3, xticks=(1:4, methods[7:end-2]), color="black", fillalpha=0.3, ylabel="Rₙ(f₀, m)", title="PID risk, n = $n")

    savefig(p1, joinpath("simulations_data", "figures", "lrr_hell_$n.pdf"))
    savefig(p2, joinpath("simulations_data", "figures", "lrr_l2_$n.pdf"))
    savefig(p3, joinpath("simulations_data", "figures", "r_pid_$n.pdf"))

end

function plot_ranks()
    methods = String[
        "Wand", "AIC", "BIC", "BR", "Knuth", "SC",
        "RIH", "RMG-B", "RMG-R", "TS", "L2CV", "KLCV"
    ]
    ns = Int64[50, 200, 1000, 5000, 25000]
    in_l2 = Bool[
        true, true, false, true, true, false, true, true,
        true, true, true, true, true, true, true, true,
    ]
    df_hell = CSV.read(joinpath("simulations_data", "hellinger_risks.csv"), DataFrame)
    df_pid = CSV.read(joinpath("simulations_data", "pid_risks.csv"), DataFrame)
    df_l2 = CSV.read(joinpath("simulations_data", "l2_risks.csv"), DataFrame)
    
    # For each row, find ranks and average them at the end
    ranks_hell = Matrix{Int64}(undef, size(df_hell, 1), size(df_hell, 2) - 2)
    ranks_pid = Matrix{Int64}(undef, size(df_pid, 1), size(df_pid, 2) - 2)
    ranks_l2 = Matrix{Int64}(undef, size(df_l2, 1), size(df_l2, 2) - 2)
    for i = 1:size(df_hell, 1)
        ranks_hell[i,:] = df_hell[i,3:end] |> Vector |> sortperm |> invperm
    end
    for i = 1:size(df_pid, 1)
        ranks_pid[i,:] = df_pid[i,3:end] |> Vector |> sortperm |> invperm
    end
    for i = 1:size(df_l2, 1)
        ranks_l2[i,:] = df_l2[i,3:end] |> Vector |> sortperm |> invperm
    end
    hell_med = mapslices(median, ranks_hell; dims=1)'
    pid_med = mapslices(median, ranks_pid; dims=1)'
    l2_med = mapslices(median, ranks_l2; dims=1)'

    p1 = plot(ylims=[0.9*minimum(hell_med), 1.1*maximum(hell_med)], ylabel="Median rank",
             xticks=(1:length(methods), methods), title="Hellinger risk")
    for j in eachindex(methods)
        plot!(p1, [j, j], [0.0, hell_med[j]], color="black", label="")
    end
    scatter!(p1, 1:length(methods), hell_med, label="", color="black", ms=6.0)
    savefig(p1, joinpath("simulations_data", "figures", "rank_hell.pdf"))

    p2 = plot(ylims=[0.9*minimum(pid_med), 1.1*maximum(pid_med)], ylabel="Median rank",
             xticks=(1:length(methods), methods), title="PID risk")
    for j in eachindex(methods)
        plot!(p2, [j, j], [0.0, pid_med[j]], color="black", label="")
    end
    scatter!(p2, 1:length(methods), pid_med, label="", color="black", ms=6.0)
    savefig(p2, joinpath("simulations_data", "figures", "rank_pid.pdf"))

    p3 = plot(ylims=[0.9*minimum(l2_med), 1.1*maximum(l2_med)], ylabel="Median ranks",
             xticks=(1:length(methods), methods), title="L2 risk")
    for j in eachindex(methods)
        plot!(p3, [j, j], [0.0, l2_med[j]], color="black", label="")
    end
    scatter!(p3, 1:length(methods), l2_med, label="", color="black", ms=6.0)
    savefig(p3, joinpath("simulations_data", "figures", "rank_l2.pdf"))
end

#= generate_risk_tables()

for n in Int64[50, 200, 1000, 5000, 25000]
    generate_lrr_figure(n)
end =#

plot_ranks()