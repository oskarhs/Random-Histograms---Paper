using RCall
@rimport ks
@rimport modetest

# Wrapper for the 'hpi' function from the 'ks'-package.
# A plug-in bandwidth for computing the derivative of the density is first computed.
# Subsequently, we compute the number of modes in the kernel estimate corresponding to this bandwidth, before the location of the modes are computed.
function hpi(x; sorted=false)
    H = rcopy(
        Tuple{Vector{Float64}, Float64},
        R"""
            x = $x

            t1 = proc.time()
            bw = ks::pmden(x)
            num_est_modes = multimode::nmodes(x, bw=bw)
            est_modes = multimode::locmodes(x, mod0=num_est_modes)
            t2 = proc.time()
            time = t2 - t1
            elapsed = unname(time['elapsed'])

            list(est_modes, elapsed)
        """
    )
    if typeof(H[1]) == Float64
        est_modes = Float64[H[1]]
    else
        est_modes = H[1]
    end
    return est_modes, dens
end