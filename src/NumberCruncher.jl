module NumberCruncher
using NearestNeighbors
include("DataManager/DataManager.jl")
using .DataManager
include("Analysis/Analysis.jl")
using .Analysis
using DynamicalSystems
using DataFrames
using Base.Threads


# Run PECUZAL embedding for a given backSize
function run(backSize; windowSize=80000,msize=10)
    # Load data and add delta columns
    mydf = NumberCruncher.DataManager.LoadData("./data/bitcoin_hourly_data_test_data.csv")
    NumberCruncher.DataManager.AddDeltas!(mydf)
    windowStart=floor(max(1,length(mydf.deltaOJittered)-backSize-windowSize))
    # Take last 'backSize' points for embedding
    dataOJit = mydf.deltaOJittered[windowStart:end-backSize]

    # Estimate delay using mutual information
    theiler = estimate_delay(dataOJit, "mi_min")

    # Perform embedding
    Y, tau_vals, ts_vals, Ls, epss = pecuzal_embedding(dataOJit; τs=0:msize, w=theiler, econ=true,verbose=false)

    tau = maximum(tau_vals)
    # Define real next point to compare with estimation
    realPoint=fill(0.0,length(tau_vals))
    c=1
    for i in reverse(tau_vals)
        realPoint[c]=mydf.deltaO[end-backSize-i+1]
        c+=1
    end
    return (Y=Y, tau_vals=tau_vals, ts_vals=ts_vals, Ls=Ls, epss=epss, DTFrame=mydf, dataOJit=dataOJit, realPoint=realPoint)
end

# Estimate the next point given a known coordinate
function analyzeit(Y, df, taus, backSize)
    # Find nearest neighbors of the last point in the embedding
    neighs = NumberCruncher.Analysis.find_neighbours(Y[begin:end-2], Y[end])
    tau = maximum(taus)
    # Compute weighted center using known X-coordinate
    rPoint=
    center = NumberCruncher.Analysis.weighted_center(
        Y[neighs.idxs .+ 1],
        neighs.dists,
        df.deltaO[end-backSize-tau+1],  # known X-coordinate
        nothing
    )
    return (center=center, neigs=neighs)
end

# Estimate the next point given a known coordinate
function analyzeit2(Y, df, taus, backSize)
    # Find nearest neighbors of the last point in the embedding
    neighs = NumberCruncher.Analysis.find_neighbours(Y[begin:end-2], Y[end])
    tau = maximum(taus)
    # Compute weighted center using known X-coordinate
    tLen=length(taus)
    rPoint::Vector{Union{Float64,Nothing}}=fill(nothing,tLen)
    reversed_taus = reverse(taus)
    for i in 1:tLen
        if(reversed_taus[i]!=0)
            rPoint[i]=df.deltaO[end-backSize-reversed_taus[i]+1]
        else
            rPoint[i]=nothing
        end
    end
    center = NumberCruncher.Analysis.weighted_center2(
        Y[neighs.idxs .+ 1],
        neighs.dists,
        rPoint
    )
    return (center=center, neigs=neighs)
end

function do_the_thing(backRange=1000:-20:10, neighRange=2:5:1000)
    nthreads = Threads.nthreads()
    # Thread-local storage for results
    results_threads = [Vector{NamedTuple}() for _ in 1:nthreads]

    # Parallel loop over backRange
    Threads.@threads for ix in 1:length(backRange)
        x = backRange[ix]
        local_results = Vector{NamedTuple}()  # local vector for this thread and iteration]
        for tausize in 5:10:100
            r0 = run(x,msize=tausize)
            println(r0.tau_vals)
            for y in neighRange
                # Make sure we don't ask for more neighbors than points
                y_actual = min(y, length(r0.Y)-1)
                Analysis.neighs = y_actual

                # Compute weighted center
                r1 = analyzeit2(r0.Y, r0.DTFrame, r0.tau_vals, x)

            # Store results in a NamedTuple
                push!(local_results, (
                    m_size=tausize,
                    time = x,
                    neighs = y_actual,
                    est_x = r1.center[1],
                    est_y = r1.center[2],
                    real_x = r0.realPoint[1],
                    real_y = r0.realPoint[2],
                ))
            end
        end
        # Append local results to thread-local storage (once per backSize)
        append!(results_threads[threadid()], local_results)
    end
    # Combine all thread-local results into a single DataFrame
    results_all = vcat(results_threads...)
    df=DataFrame(results_all)
    df.dists=sqrt.((df.est_x .- df.real_x).^2 .+ (df.est_y .- df.real_y).^2)
    return df
end
using PlotlyJS

function plotit(r)
    plt = Plot(
        scatter3d(
            x = r.size,
            y = r.neighs,
            z = r.dists,
            mode = "markers",
            marker = attr(
                size = 2,
                color = r.dist,
                colorscale = "Viridis",
                showscale = true
            ),
            name = "Distance points"  # this will appear in legend
        ),
        Layout(
            title = "3D Scatter Plot",
            scene = attr(
                xaxis = attr(title="Time"),
                yaxis = attr(title="Neighbours"),
                zaxis = attr(title="Distance")
            ),
            showlegend = true
        )
    )

    display(plt)
end




end
