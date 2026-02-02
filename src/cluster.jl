# src/cluster.jl
import ..model_gf2
import ..his_gf


function obj_func(ps, sel, Xl, PF::Vector, PF2::AbstractMatrix,
                  z::Vector, wll::Vector, γn::Vector, hi::Vector)
    θ = [vcat(ps[1]*Xl[i], ps[2:sel]) for i in eachindex(Xl)]
    U = [ (x->model_gf2(θ[i], x, sel)).(z) for i in eachindex(Xl) ]
    U = hcat(U...)
    μ_2 = vec(mean(U.^2, weights(PF), dims=2))
    μ   = [mean(U[i, :], weights(PF2[i, :])) for i in 1:length(z)] .* hi
    res =  mean(γn) * wll' * μ_2 - 2 * wll' * μ
    return res
end

function infer_theta_ini(rd_gf_p, γ, sel, total_rd_no, z, wll, k)
    res = zeros(sel*k)
    PF2 = zeros(length(z), 13)
    for nti in 1:k
        β_1 = vec(total_rd_no)
        xl, wl = gausslegendre(13)

        Density1 = kde(β_1, weights=weights(γ[:, nti]))
        min_β, max_β = minimum(β_1), maximum(β_1)
        Xl = (max_β-min_β)/2 .* xl .+ (max_β+min_β)/2
        ik1 = InterpKDE(Density1)
        pf1 = pdf(ik1, Xl)
        PF  = pf1 .* wl .* (max_β-min_β)/2

        wei = rd_gf_p .* γ[:, nti]'
        for i in 1:length(z)
            Density = kde(β_1, weights=weights(wei[i, :]))
            ik = InterpKDE(Density)
            pf = pdf(ik, Xl)
            PF2[i, :] = pf .* wl .* (max_β-min_β)/2
        end

        hi = vec(mean(rd_gf_p .* γ[:, nti]', dims=2))

        init_ps = zeros(sel)
        out = optimize(
            ps -> obj_func(exp.(ps), sel, Xl, PF, PF2, z, wll, γ[:, nti], hi),
            init_ps,
            Optim.Options(show_trace=false, g_tol=1e-9, iterations=800)
        )
        params = exp.(out.minimizer)
        res[(nti-1)*sel+1 : nti*sel] .= params
    end
    return res
end

function compute_dist(resul, rd_gf_p, wl, k, n, sel, z, β)
    dist2 = zeros(n, k)
    for j = 1:k
        resul_tmp = [vcat(resul[(j-1)*sel+1] * β[i], resul[(j-1)*sel+2:j*sel]) for i = 1:n]
        θ_j = hcat([(x -> model_gf2(resul_tmp[i], x, sel)).(z) for i = 1:n]...)
        dist2[:, j] = (wl' * (rd_gf_p .- θ_j).^2)
    end
    return dist2
end

function prime_cluster(rdn, rdm, β, k::Int;
                       sel::Int = 4,
                       maxiter::Int = 20,
                       z_min::Float64 = 0.95,
                       z_max::Float64 = 1.0,
                       n_z::Int = 7)

    n, p = size(rdm)
    @assert size(rdn) == (n, p) 

    xl, wl = gausslegendre(n_z)
    zo = (z_max - z_min)/2 .* xl .+ (z_max + z_min)/2
    z  = [(zo[i], zo[j]) for i in 1:length(zo), j in 1:length(zo)]
    z  = vec(z)
    wl = wl * (z_max - z_min)/2
    wl = vec(wl * wl')

    rd_gf = SharedArray{Float64}(length(z), n, p)
    @sync @distributed for gj in 1:p
        for i in 1:n
            rd_gf[:, i, gj] = (x -> his_gf(rdn[i, gj], rdm[i, gj], x)).(z)
        end
    end

    X = rdn + rdm
    R = kmeans(X', k)
    μ_cen = R.centers
    prob_ini = zeros(n, k)
    for j in 1:k
        di = map(i -> -euclidean(μ_cen[:, j], X[i, :]), 1:n)
        prob_ini[:, j] .= di
    end

    γ1 = rand(Uniform(0.1, 0.6), n, k)
    for i in 1:n
        _, idx = findmax(prob_ini[i, :])
        γ1[i, idx] = 0.9
    end
    γ = γ1 ./ sum(γ1, dims=2)

    s = -1.0
    tot_old = Inf

    for iter in 1:maxiter
        results_θ_t = pmap(gj -> infer_theta_ini(rd_gf[:, :, gj], γ, sel, β, z, wl, k), 1:p)
        results_θ  = Matrix(hcat(results_θ_t...)')   # p × (sel*k)

        results_dist = pmap(gj -> compute_dist(results_θ[gj, :], rd_gf[:, :, gj], wl, k, n, sel, z, β), 1:p)
        D = sum(results_dist) 

        dij   = (1/k) .* D .^ (s - 1)
        normd = (1/k) .* sum(D .^ s, dims=2)
        replace!(dij,   Inf => prevfloat(Inf))
        replace!(normd, Inf => prevfloat(Inf))
        γ = dij .* normd.^(1/s - 1)
        tot = sum(D .* γ)
        tot_old = tot
        @info "PRIME iter $iter, obj = $tot"
        if s > -56
            s *= 1.5
        end
    end

    cluster_labels = map(argmax, eachrow(γ)) |> collect

    return cluster_labels, γ
end
