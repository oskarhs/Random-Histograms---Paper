using AutoHist, Plots, DataFrames, CSV, Latexify, StatsPlots
import Statistics: median

function generate_risk_tables()
    densities = [1, 2, 4, 7, 8, 11, 13, 14]
    methods = ["RIH", "SI", "FM", "WJ"]
    ns = Int64[200, 1000, 5000]
    df_pid = CSV.read(joinpath(@__DIR__, "simulations_data", "pid_risks.csv"), DataFrame)
    df_elapsed = CSV.read(joinpath(@__DIR__, "simulations_data", "elapsed.csv"), DataFrame)

    # Convert some columns to integer
    df_pid[!, :n] = convert.(Int64, df_pid[!, :n])
    df_pid[!, :Density] = convert.(Int64, df_pid[!, :Density])

    # First, create all risk tables to be written to file
    colnms = vcat(["Density"], ["n"], methods)
    df_pid_print_1 = DataFrame([[] for _ = colnms] , colnms)

    for j in 1:8
        for i in eachindex(ns)
            # Need to alternate between pid risk and elapsed
            push!(df_pid_print_1, df_pid[3*(j-1)+i,:])
            if i != 1
                df_pid_print_1[end,1] = ""
            end
            df_pid_print_1[end,3:end] = round.(values(df_pid_print_1[end,3:end]); digits=3)
        end
    end

#=     \begin{tabular}{cccccccccccccc}
    \toprule
    Density & n & Wand & AIC & BIC & BR & Knuth & SC & RIH & RMG-B & RMG-R & TS & L2CV & KLCV\\
    \midrule =#

    # Insert new line at index 4

    # Pid table part 1
    j = 4
    pid_table = latexify(df_pid_print_1; env = :table, booktabs = true, latex = false)
    pid_table_vec = split(pid_table, "\n")
    for i in 1:7
        pid_table_vec[j+3*i] = pid_table_vec[j+3*i] * " \\hline" 
    end
    # Insert new line specifying which columns show risk and which show median elapsed time
    insert!(pid_table_vec, 4, " & " * repeat("& PID risk & Median t", length(methods)) * "\\midrule")
    pid_table = join(pid_table_vec, "\n")
    open(joinpath(@__DIR__, "simulations_data", "risk_tables", "pid_risk_table_1.txt"), "w") do io
        println(io, pid_table)
    end
end

generate_risk_tables() # This has to be ediate