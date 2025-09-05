module Analysis
using Statistics


using NearestNeighbors
neighs=30

function weighted_center(points, dists,cx,cy)
    X = [p[1] for p in points]
    Y = [p[2] for p in points]
    m=minimum(dists)
    if cx !== nothing
        dists .+= ((X .- cx).^4)
    end
    if cy !== nothing
        dists .+= ((Y .- cy).^4)
    end

    weights = m ./ dists
    
    return [sum(weights .* X) / sum(weights), sum(weights .* Y) / sum(weights)]
end

function find_neighbours(spaceSet,point)
    tree=KDTree(spaceSet)
    idxes,dists=knn(tree,point,neighs)
    return (idxs=idxes,dists=dists,NSet=spaceSet[idxes],tree=tree)
end




end