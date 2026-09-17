module Analysis
using Statistics


using NearestNeighbors
neighs=30

function weighted_center(points, dists,cx,cy)
    X = [p[1] for p in points]
    Y = [p[2] for p in points]
    m=minimum(dists)
    if cx !== nothing
        dists .+= ((X .- cx).^2)
    end
    if cy !== nothing
        dists .+= ((Y .- cy).^2)
    end
 

    weights = m ./ dists
    
    return [sum(weights .* X) / sum(weights), sum(weights .* Y) / sum(weights)]
end

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
   # weights
    w = 1 ./ eff_dists
    denom = sum(w)
    M = reduce(hcat, points)
    for j in 1:d
        result[j] = sum(points[:,j] .* w) / denom 
    end
    return result
end




function find_neighbours(spaceSet,point)
    tree=KDTree(spaceSet)
    idxes,dists=knn(tree,point,neighs)
    return (idxs=idxes,dists=dists,NSet=spaceSet[idxes],tree=tree)
end




end