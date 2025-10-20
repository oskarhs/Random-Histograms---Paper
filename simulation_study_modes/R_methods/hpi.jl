using RCall
@rimport ks
@rimport LPCM

# Wrapper for the 'hpi' function from the 'ks'-package.
# A plug-in bandwidth for computing the derivative of the density is first computed.
# Subsequently, we compute the number of modes in the kernel estimate corresponding to this bandwidth, before the location of the modes are computed.
function hpi(x)
    obj = rcopy(
        R"""
            x = $x

            t1 = proc.time()
            bw = ks::hpi(x, deriv.order=1) # Determine plug-in bandwidth
            est_modes = as.vector(LPCM::ms(x, bw, scaled=0)[['cluster.center']]) # Use chosen bandwidth to estimate mode locations
            t2 = proc.time()
            time = t2 - t1
            elapsed = unname(time['elapsed'])

            list(est_modes=est_modes, elapsed=elapsed)
        """
    )
    tup = obj[:est_modes], obj[:elapsed]
    if typeof(tup[1]) == Float64
        est_modes = Float64[tup[1]]
    elseif typeof(tup[1]) == Missing
        est_modes = Missing[missing]
    else
        est_modes = tup[1]
    end
    elapsed = tup[2]
    return est_modes, elapsed
end