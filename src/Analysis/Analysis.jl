module Analysis
using Statistics


using NearestNeighbors
# Default k for the k-NN search; always passed explicitly, never mutated (thread safety).
const DEFAULT_K = 30

"""
Weights of the neighbours' successors: 1 / (embedding distance + distance to the known
coordinates of the next point). shadow=false drops the second term (plain k-NN analogues).
"""
function neighbour_weights(points, dists::Vector{Float64}, rPoint::Vector{Union{Float64,Nothing}};
                           shadow::Bool=true)
    d = length(rPoint)
    # effective distances
    eff_dists = copy(dists)
   # known coordinates added to weights
    if shadow
        for i in 1:length(dists)
            distance=0.0
            for j in 1:d
                if(rPoint[j]!=nothing)
                   distance+=(rPoint[j]-points[i][j])^2
                end
            end
            eff_dists[i]+=sqrt(distance)
        end
    end
   # weights; ε keeps an exact match (distance 0) from giving an infinite weight
    return 1 ./ (eff_dists .+ 1e-12 * max(1.0, maximum(eff_dists)))
end

# Weighted mean of f(coordinate j) over the successors.
weighted_coordinate(points, w, j; f=identity) = sum(f.(points[:,j]) .* w) / sum(w)

function weighted_center2(points,
                         dists::Vector{Float64},
                         rPoint::Vector{Union{Float64,Nothing}}; shadow::Bool=true)
    w = neighbour_weights(points, dists, rPoint; shadow=shadow)
    return [weighted_coordinate(points, w, j) for j in 1:length(rPoint)]
end




function find_neighbours(spaceSet,point,k=DEFAULT_K)
    tree=KDTree(spaceSet)
    idxes,dists=knn(tree,point,k)
    return (idxs=idxes,dists=dists,NSet=spaceSet[idxes],tree=tree)
end

"""
Scores |r| forecasts: MAE and skill against the EWMA forecast σ·√(2/π) and persistence |r_t|,
the correlation, and the share of points where the model's error is below the EWMA's
(the hit rate is meaningless for |r|).
"""
function abs_skill_summary(est::AbstractVector, real::AbstractVector, last::AbstractVector,
                           ewma::AbstractVector)
    mae(e) = mean(abs.(e .- real))
    return (n=length(real), mae=mae(est), mae_ewma=mae(ewma), mae_persistence=mae(last),
            skill_ewma=1 - mae(est) / mae(ewma), skill_persistence=1 - mae(est) / mae(last),
            corr=cor(est, real), corr_ewma=cor(ewma, real),
            beats_ewma=mean(abs.(est .- real) .< abs.(ewma .- real)))
end

"""
Two-sided exact binomial test p-value for `hits` successes in `n` trials with p = 0.5
"""
function binomial_pvalue(hits::Integer, n::Integer)
    n == 0 && return 1.0
    t = min(hits, n - hits)
    tail = sum(binomial(big(n), j) for j in 0:t) / big(2)^n
    return Float64(min(1, 2 * tail))
end

"""
AR(1) forecast of the value after `last`, fitted on `window` (mean and lag-1 autocorrelation).
A fair baseline for smoothed series: differences of daily/weekly means are autocorrelated
(lag-1 ≈ 0.25 even for a pure random walk), which this baseline already captures.
"""
function ar1_forecast(window::AbstractVector{<:Real}, last::Real)
    μ = mean(window)
    φ = length(window) > 2 ? cor(@view(window[1:end-1]), @view(window[2:end])) : 0.0
    return μ + (isfinite(φ) ? φ : 0.0) * (last - μ)
end

"""
Scores estimates of a series of changes against naive baselines

Parameters:
- est, real: estimated and real values
- last: the last known value before each estimated one (the persistence forecast)
- ar1: optional AR(1) forecasts (see ar1_forecast) to score against as well

Returns:
- NamedTuple with MAE of the model, of the zero predictor and of persistence, the skills
  1 - MAE/MAE_baseline, the hit rate of the sign (and of the sign of the last change),
  and the binomial p-value of the hit rate against a coin flip
"""
function skill_summary(est::AbstractVector, real::AbstractVector, last::AbstractVector;
                       ar1::Union{AbstractVector,Nothing}=nothing)
    mae(e) = mean(abs.(e .- real))
    ok = sign.(real) .!= 0
    hits = count(sign.(est[ok]) .== sign.(real[ok]))
    n = count(ok)
    base = (n=length(real), mae=mae(est), mae_zero=mae(zero(real)), mae_persistence=mae(last),
            skill_zero=1 - mae(est) / mae(zero(real)), skill_persistence=1 - mae(est) / mae(last),
            hit_rate=n == 0 ? NaN : hits / n,
            hit_rate_last_sign=n == 0 ? NaN : count(sign.(last[ok]) .== sign.(real[ok])) / n,
            hit_pvalue=binomial_pvalue(hits, n))
    ar1 === nothing && return base
    return merge(base, (mae_ar1=mae(ar1), skill_ar1=1 - mae(est) / mae(ar1)))
end




end