using RCall
@rimport histogram as rmg

function rmg_hist(x, penalty)
    H = rcopy(
        Tuple{Vector{Float64}, Union{Float64, Vector{Float64}}},
        R"""
            h = histogram::histogram($x, type="irregular", verbose=FALSE, plot=FALSE, penalty=$penalty)
            
            brks = h$breaks
            dens = h$density
            list(brks, dens)
        """
    )
    if typeof(H[2]) == Float64
        dens = Float64[H[2]]
    else
        dens = H[2]
    end
    return H[1], dens
end