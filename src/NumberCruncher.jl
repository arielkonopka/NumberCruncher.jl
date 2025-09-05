module NumberCruncher
using NearestNeighbors
include("DataManager/DataManager.jl")
using .DataManager
include("Analysis/Analysis.jl")
using .Analysis
using DynamicalSystems
using DataFrames
using Base.Threads

msize = 50  # maximum embedding dimension

# Run PECUZAL embedding for a given backSize
function run(backSize)
    # Load data and add delta columns
    mydf = NumberCruncher.DataManager.LoadData("./data/bitcoin_hourly_data_test_data.csv")
    NumberCruncher.DataManager.AddDeltas!(mydf)

    # Take last 'backSize' points for embedding
    dataOJit = mydf.deltaOJittered[begin:end-backSize]

    # Estimate delay using mutual information
    theiler = estimate_delay(dataOJit, "mi_min")

    # Perform embedding
    Y, tau_vals, ts_vals, Ls, epss = pecuzal_embedding(dataOJit; τs=0:msize, w=theiler, econ=true)

    tau = maximum(tau_vals)
    # Define real next point to compare with estimation
    realPoint = [mydf.deltaO[end-backSize-tau+1], mydf.deltaO[end-backSize+1]]

    return (Y=Y, tau_vals=tau_vals, ts_vals=ts_vals, Ls=Ls, epss=epss, DTFrame=mydf, dataOJit=dataOJit, realPoint=realPoint)
end

# Estimate the next point given a known coordinate
function analyzeit(Y, df, taus, backSize)
    # Find nearest neighbors of the last point in the embedding
    neighs = NumberCruncher.Analysis.find_neighbours(Y[begin:end-2], Y[end])
    tau = maximum(taus)
    # Compute weighted center using known X-coordinate
    center = NumberCruncher.Analysis.weighted_center(
        Y[neighs.idxs .+ 1],
        neighs.dists,
        df.deltaO[end-backSize-tau+1],  # known X-coordinate
        nothing
    )
    return (center=center, neigs=neighs)
end

function do_the_thing(backRange=1000:-15:10, neighRange=2:3:1000)
    nthreads = Threads.nthreads()
    # Thread-local storage for results
    results_threads = [Vector{NamedTuple}() for _ in 1:nthreads]

    # Parallel loop over backRange
    Threads.@threads for ix in 1:length(backRange)
        x = backRange[ix]
        local_results = Vector{NamedTuple}()  # local vector for this thread and iteration
        r0 = run(x)

        for y in neighRange
            # Make sure we don't ask for more neighbors than points
            y_actual = min(y, length(r0.Y)-1)
            Analysis.neighs = y_actual

            # Compute weighted center
            r1 = analyzeit(r0.Y, r0.DTFrame, r0.tau_vals, x)

            # Store results in a NamedTuple
            push!(local_results, (
                size = x,
                neighs = y_actual,
                est_x = r1.center[1],
                est_y = r1.center[2],
                real_x = r0.realPoint[1],
                real_y = r0.realPoint[2]
            ))
        end

        # Append local results to thread-local storage (once per backSize)
        append!(results_threads[threadid()], local_results)
    end

    # Combine all thread-local results into a single DataFrame
    results_all = vcat(results_threads...)
    return DataFrame(results_all)
end

end
