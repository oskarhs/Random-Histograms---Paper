using AutoHist, Plots

p_values = parse.(Float64, readlines("hedenfalk.txt"))

H1, _ = histogram_irregular(p_values; grid="regular", support=(0.0, 1.0))
plot(H1, alpha=0.5)

H2, _ = histogram_regular(p_values; a = k->0.5*k)
plot(H2, alpha=0.5)
