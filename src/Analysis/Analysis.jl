module Analysis
using Statistics


using NearestNeighbors
# Default k for the k-NN search; always passed explicitly, never mutated (thread safety).
const DEFAULT_K = 30

function weighted_center2(points,
                         dists::Vector{Float64},
                         rPoint::Vector{Union{Float64,Nothing}})
    d = length(rPoint)
    n = length(points)
    result = Vector{Float64}(undef, d)

    # effective distances
    eff_dists = copy(dists)
   # known coordinates added to weights
    for i in 1:length(dists)
        distance=0.0
        for j in 1:d
            if(rPoint[j]!=nothing)
               distance+=(rPoint[j]-points[i][j])^2
            end
        end
        eff_dists[i]+=sqrt(distance)
    end
   # weights; ε keeps an exact match (distance 0) from giving an infinite weight
    w = 1 ./ (eff_dists .+ 1e-12 * max(1.0, maximum(eff_dists)))
    denom = sum(w)
    for j in 1:d
        result[j] = sum(points[:,j] .* w) / denom 
    end
    return result
end




function find_neighbours(spaceSet,point,k=DEFAULT_K)
    tree=KDTree(spaceSet)
    idxes,dists=knn(tree,point,k)
    return (idxs=idxes,dists=dists,NSet=spaceSet[idxes],tree=tree)
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
Scores estimates of a series of changes against naive baselines

Parameters:
- est, real: estimated and real values
- last: the last known value before each estimated one (the persistence forecast)

Returns:
- NamedTuple with MAE of the model, of the zero predictor and of persistence, the skills
  1 - MAE/MAE_baseline, the hit rate of the sign (and of the sign of the last change),
  and the binomial p-value of the hit rate against a coin flip
"""
function skill_summary(est::AbstractVector, real::AbstractVector, last::AbstractVector)
    mae(e) = mean(abs.(e .- real))
    ok = sign.(real) .!= 0
    hits = count(sign.(est[ok]) .== sign.(real[ok]))
    n = count(ok)
    return (n=length(real), mae=mae(est), mae_zero=mae(zero(real)), mae_persistence=mae(last),
            skill_zero=1 - mae(est) / mae(zero(real)), skill_persistence=1 - mae(est) / mae(last),
            hit_rate=n == 0 ? NaN : hits / n,
            hit_rate_last_sign=n == 0 ? NaN : count(sign.(last[ok]) .== sign.(real[ok])) / n,
            hit_pvalue=binomial_pvalue(hits, n))
end




end