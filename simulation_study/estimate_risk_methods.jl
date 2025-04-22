using Random

include("loss_functions.jl")

function get_methods()
    return [
        "Wand", "AIC", "BIC", "BR", "Knuth", "SC",
        "RIH", "RMG-B", "RMG-R", "TS", "L2CV", "KLCV"
    ]
end

# Compute bins on a regular grid
function bin_regular(x::AbstractVector{<:Real}, xmin::Real, xmax::Real, k::Int, right::Bool)
    R = xmax - xmin
    bincounts = zeros(Float64, k)
    edges_inc = k/R
    if right
        for val in x
            idval = min(k-1, floor(Int, (val-xmin)*edges_inc+eps())) + 1
            @inbounds bincounts[idval] += 1.0
        end
    else
        for val in x
            idval = max(0, floor(Int, (val-xmin)*edges_inc-eps())) + 1
            @inbounds bincounts[idval] += 1.0
        end
    end
    return bincounts
end

# Estimate risk for all methods for a sample of size n from a density f₀
function estimate_risk(rng, d, n, B; l2=false)
    methods = get_methods()
    num_methods = length(methods)    

    risk_hell = Array{Float64}(undef, num_methods)
    risk_l2 = Array{Float64}(undef, num_methods)
    risk_pid = Array{Float64}(undef, num_methods)
    loss_hell = Array{Float64}(undef, B, num_methods)
    loss_l2 = Array{Float64}(undef, B, num_methods)
    loss_pid = Array{Float64}(undef, B, num_methods)

    for b = 1:B
        x = rand(rng, d, n)
        xmin = minimum(x)
        xmax = maximum(x)
        R = xmax - xmin
        h_wand = rcopy(Float64, R"suppressWarnings(KernSmooth::dpih($x, level=2L))")
        k = ceil(Int64, R/h_wand) # Wand, twostep
        N = bin_regular(x, xmin, xmax, k, true)
        breaks = collect(LinRange(xmin, xmax, k+1))
        loss_hell[b,1] = hellinger_loss(d, breaks, k*N/(n*R))
        loss_pid[b,1], _, _, _ = peak_id_loss(d, breaks, k*N/(n*R))
        if l2
            loss_l2[b,1] = l2_loss(d, breaks, k*N/(n*R))
        end
        for j = 2:4
            # aic, bic and br
            H, _ = histogram_regular(x; rule=methods[j])
            breaks = collect(H.edges[1])
            loss_hell[b,j] = hellinger_loss(d, breaks, H.weights)
            loss_pid[b,j], _, _, _ = peak_id_loss(d, breaks, H.weights)
            if l2
                loss_l2[b,j] = l2_loss(d, breaks, H.weights)
            end
        end
        # Knuth
        H, _ = histogram_regular(x; rule="bayes", a = a->0.5*k)
        breaks = collect(H.edges[1])
        loss_hell[b,5] = hellinger_loss(d, breaks, H.weights)
        loss_pid[b,5], _, _, _ = peak_id_loss(d, breaks, H.weights)
        if l2
            loss_l2[b,5] = l2_loss(d, breaks, H.weights)
        end
        # SC
        H, _ = histogram_regular(x; rule="bayes", a = a->1.0*k)
        breaks = collect(H.edges[1])
        loss_hell[b,6] = hellinger_loss(d, breaks, H.weights)
        loss_pid[b,6], _, _, _ = peak_id_loss(d, breaks, H.weights)
        if l2
            loss_l2[b,6] = l2_loss(d, breaks, H.weights)
        end
    end
    for j=1:num_methods
        risk_hell[j] = mean(loss_hell[:,j])
        risk_pid[j] = mean(loss_pid[:,j])
        risk_l2[j] = mean(loss_l2[:,j])
    end
    return risk_hell, risk_pid, risk_l2
end

@time estimate_risk(Xoshiro(1812), Normal(), 25000, 500; l2=true)