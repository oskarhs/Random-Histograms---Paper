using RCall
@rimport KernSmooth

function wand_hist(x)
    n = length(x)
    xmin = minimum(x)
    xmax = maximum(x)
    R = xmax - xmin
    h_wand = rcopy(Float64, R"suppressWarnings(KernSmooth::dpih($x, level=2L))")
    k = ceil(Int64, R/h_wand) # Wand, twostep
    N = bin_regular(x, xmin, xmax, k, true)
    breaks = collect(LinRange(xmin, xmax, k+1))
    return breaks, k*N/(n*R)
end