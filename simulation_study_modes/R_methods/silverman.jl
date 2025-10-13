using RCall
@rimport multimode

# Wrapper for the 'pmden' function from the 'ftnonpar'-package of Davies and Kovac (2012)
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
        est_modes = suppressWarnings(multimode::locmodes(x, mod0=num_est_modes, n=1024)[['locations']])
        t2 = proc.time()
        time = t2 - t1
        elapsed = unname(time['elapsed'])
        list(est_modes=est_modes, elapsed=elapsed)
    """
    tup = rcopy(Tuple{Union{Float64, Vector{Float64}}, Float64}, robj)
    if typeof(tup[1]) == Float64
        est_modes = Float64[tup[1]]
    else
        est_modes = tup[1]
    end
    elapsed = tup[2]
    return est_modes, elapsed
end