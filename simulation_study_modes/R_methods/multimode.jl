using RCall
@rimport multimode

# Wrapper for the 'modetest' function from the 'multimode'-package
# Method should either be 'SI' or 'FM'.
function modetest(x::Vector{Float64}; method::String)
    robj = R"""
        x = $x
        method = $method
        alpha = 0.1
        num_est_modes = 0
        p_value = 0.0
        iter = 1
        maxiter = 10
        t1 = proc.time()
        while (p_value <= alpha && iter <= maxiter){
            num_est_modes = num_est_modes + 1
            p_value = multimode::modetest(x, mod0=num_est_modes, method=method, n=1024, B=100)[['p.value']]
            iter = iter + 1
        }
        # Note: Odd indices are modes, even indices are antimodes
        extremes = suppressWarnings(multimode::locmodes(x, mod0=num_est_modes, n=1024))[['locations']]
        est_modes = extremes[seq(1,length(extremes), by=2)] # Take length of est_modes, not obj.
        t2 = proc.time()
        time = t2 - t1
        elapsed = unname(time['elapsed'])
        list(est_modes=est_modes, elapsed=elapsed)
    """
    obj = rcopy(robj)
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