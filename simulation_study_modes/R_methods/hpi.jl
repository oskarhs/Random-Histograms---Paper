using RCall
@rimport ks
@rimport multimode

# Wrapper for the 'hpi' function from the 'ks'-package.
# A plug-in bandwidth for computing the derivative of the density is first computed.
# Subsequently, we compute the number of modes in the kernel estimate corresponding to this bandwidth, before the location of the modes are computed.
function hpi(x)
    H = rcopy(
        Tuple{Union{Float64, Vector{Float64}}, Float64},
        R"""
            x = $x

            t1 = proc.time()
            bw = ks::hpi(x, deriv.order=1)
            num_est_modes = suppressWarnings(multimode::nmodes(x, bw=bw))
            # Note: Odd indices are modes, even indices are antimodes
            extremes = suppressWarnings(multimode::locmodes(x, mod0=num_est_modes, n=1024))[['locations']]
            est_modes = extremes[seq(1,length(extremes), by=2)] # Take length of est_modes, not obj.
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
    return est_modes, H[2]
end