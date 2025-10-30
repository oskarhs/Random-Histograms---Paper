using RCall
@rimport mclust
@rimport BayesMultiMode

# Wrapper for the 'Mclust' function from the 'mclust'-package for estimating a Gaussian mixture model,
# combined with the modal EM algorithm for finding the resulting modes of the mixture.
function gmm(x::Vector{Float64})
    H = rcopy(
        Tuple{Union{Float64, Vector{Float64}}, Float64},
        R"""
            x = $x
            min_x = min(x)
            max_x = max(x)

            t1 = proc.time()

            # Estimate a Gaussian mixture model, select number of components/variance structure via BIC
            fit_gmm = mclust::Mclust(x, G=1:10)

            # Extract mixture parameters 
            mu = fit_gmm$parameters$mean
            vars = fit_gmm$parameters$variance$sigmasq
            if (length(vars) == 1) {
                sigma = rep(sqrt(vars), length(mu))
            } else {
                sigma = sqrt(vars)
            }
            eta = fit_gmm$parameters$pro
            params = c(eta = eta, mu = mu, sigma = sigma)
            dist = "normal"
            
            # Create a mixture object and run the modal EM algorithm
            mix = BayesMultiMode::mixture(params, dist = dist, range = c(min_x,max_x))
            est_modes = BayesMultiMode::mix_mode(mix)['mode_estimates'][[1]]

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