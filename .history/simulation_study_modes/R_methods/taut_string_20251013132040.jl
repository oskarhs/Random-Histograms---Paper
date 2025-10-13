using RCall
@rimport ftnonpar

# Wrapper for the 'pmden' function from the 'ftnonpar'-package of Davies and Kovac (2012)
function taut_string(x; sorted=false)
    if !sorted
        sort!(x)
    end
    H = rcopy(
        Tuple{Vector{Float64}, Union{Float64, Vector{Float64}}, Float64},
        R"""
            x = $x

            t1 = proc.time()
            h = ftnonpar::pmden(x)
            ind = h$ind
            
            brks = x[ind]
            dens = h$y[ind[1:(length(ind)-1)]]
            t2 = proc.time()
            time = t2 - t1
            elapsed = unname(time['elapsed'])

            list(brks, dens, elapsed)
        """
    )
    if typeof(H[2]) == Float64
        dens = Float64[H[2]]
    else
        dens = H[2]
    end
    return H[1], dens, H[3]
end