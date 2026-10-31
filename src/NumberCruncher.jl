module NumberCruncher
using NearestNeighbors
include("DataManager/DataManager.jl")
using .DataManager
include("Analysis/Analysis.jl")
using .Analysis
include("Preprocessing/Preprocessor.jl")
using .Preprocessor
using DynamicalSystems
using DataFrames
using Base.Threads
using HTTP
using JSON
using CSV
using Dates
using Random
using Statistics

# `run` is deliberately not exported: it would clash with Base.run.
export reconstruct, run_series, analyzeit2, do_the_thing, do_the_thing2, summarize,
       update_from_cryptocompare!, plot_aesthetic_analysis, serve_gui
const CRYPTOCOMPARE_KEY_FILE = joinpath(@__DIR__, "..", "config", "cryptocompare_key.txt")
# Read at call time: the key is only needed for downloads, not to load the package.
# The CRYPTOCOMPARE_KEY environment variable takes precedence over the file.
function cryptocompare_key()
    key = strip(get(ENV, "CRYPTOCOMPARE_KEY", ""))
    isempty(key) || return key
    isfile(CRYPTOCOMPARE_KEY_FILE) ||
        error("missing API key: set CRYPTOCOMPARE_KEY or put it in $CRYPTOCOMPARE_KEY_FILE")
    return strip(read(CRYPTOCOMPARE_KEY_FILE, String))
end

# ==============================================================================
# Data Fetch (CRYPTOCOMPARE - PORT Z PYTHONA)
# ==============================================================================
"""
Downloades historical data from z CryptoCompare, from start_date to current time.
Returns (ok, rows, message); `message` holds the API error if the download stopped early.
"""
function update_from_cryptocompare!(filepath="./data/btc_cryptocompare.csv"; start_date=Dates.DateTime(2008, 1, 1))
    symbol = "BTC"
 #   symbol = "ETH"
    limit = 2000 # Max dla CryptoCompare
    
    # convert dates to timestamps (seconds)
    start_ts = Int64(floor(Dates.datetime2unix(start_date)))
    end_ts = Int64(floor(Dates.datetime2unix(Dates.now())))
    
    println("Fetching data to: $filepath")
    
    all_dfs = DataFrames.DataFrame[]
    current_end_ts = end_ts
    api_error = nothing
    
    try
        api_key = cryptocompare_key()
        while current_end_ts > start_ts
            url = "https://min-api.cryptocompare.com/data/v2/histohour"
            params = [
                "fsym" => symbol,
                "tsym" => "USD",
                "limit" => string(limit),
                "toTs" => string(current_end_ts),
            ]
            
            # Wysłanie zapytania; the key goes in a header so it stays out of URL logs.
            query_str = join(["$k=$v" for (k,v) in params], "&")
            response = HTTP.get(url * "?" * query_str, ["authorization" => "Apikey $api_key"])
            data = JSON.parse(String(response.body))
            
            if data["Response"] != "Success"
                println("Błąd API: $(data["Message"])")
                api_error = data["Message"]
                break
            end
            
            raw_list = data["Data"]["Data"]
            if isempty(raw_list)
                println("Brak więcej danych.")
                break
            end
            
            # Mapowanie pól (CryptoCompare używa 'time' jako timestamp w sekundach)
            batch_df = DataFrames.DataFrame(
                timestamp = Int64[Int64(d["time"]) * 1000 for d in raw_list], # Konwersja na ms dla kompatybilności
                open = Float64[Float64(d["open"]) for d in raw_list],
                high = Float64[Float64(d["high"]) for d in raw_list],
                low = Float64[Float64(d["low"]) for d in raw_list],
                close = Float64[Float64(d["close"]) for d in raw_list],
                volume = Float64[Float64(d["volumefrom"]) for d in raw_list]
            )
            filter!(row -> row.close > 0.0, batch_df)
	    if isempty(batch_df)
                println("Osiągnięto limit historycznych danych (same zera). Przerywam.")
                break
            end
            pushfirst!(all_dfs, batch_df)
            
            # Aktualizacja timestampu (najstarszy z paczki minus 1 sekunda)
            earliest_in_batch = raw_list[1]["time"]
            current_end_ts = earliest_in_batch - 1
            
            println("Pobrano dane do: $(Dates.unix2datetime(earliest_in_batch))")
            
            sleep(0.5) # Respektowanie limitów API
            
            if current_end_ts < start_ts
                break
            end
        end
        
        if !isempty(all_dfs)
            final_df = vcat(all_dfs...)
            # Usuń duplikaty jeśli istnieją
            unique!(final_df, :timestamp)
            
            mkpath(dirname(filepath))
            CSV.write(filepath, final_df)
            println("Sukces! Zapisano $(size(final_df, 1)) rekordów do $filepath")
            return (ok=true, rows=size(final_df, 1), message=something(api_error, "complete"))
        end
        return (ok=false, rows=0, message=something(api_error, "no data received"))
    catch e
        println("Wystąpił błąd: $e")
        return (ok=false, rows=0, message=sprint(showerror, e))
    end
end








# ==============================================================================
# PHASE SPACE RECONSTRUCTION (PECUZAL)
# ==============================================================================
# Funkcja przygotowuje manifest (przestrzeń fazową) dla konkretnego punktu w czasie.
# 'backSize' określa przesunięcie od końca danych w celu przeprowadzenia testu.
# Works on any prepared series s (price deltas, a Lorenz coordinate, ...).
# Pass a seeded `rng` (e.g. Xoshiro(seed)) to make the jitter, and so the result, reproducible.
# `jitter` is relative to the window's std: quantised data (prices in cents, pressure in 0.1 hPa)
# has exact duplicates, and an absolute 1e-10 is too small for PECUZAL on values near 1000.
#
# samplesize < 1 lets PECUZAL evaluate only that fraction of points (much faster, but it can pick
# other delays); its subsample is seeded from `rng`. `embedding` = (tau_vals, theiler) skips
# PECUZAL and only builds the delay vectors of this window (see EmbeddingBlocks).
function run_series(s::AbstractVector{<:Real}, backSize; windowSize=82000, msize=10, jitter=1e-8,
                    rng::AbstractRNG=Random.default_rng(), samplesize::Real=1.0, embedding=nothing)
    # Określenie okna czasowego obserwacji.
    windowStart = max(1, length(s) - backSize - windowSize)
    window = s[windowStart:end-backSize]
    # Dodawanie jittera (szumu zapobiegającego błędom k-NN).
    scale = std(window) > 0 ? std(window) : 1.0
    dataOJit = window .+ (jitter * scale) .* randn(rng, length(window))

    if embedding === nothing
        # Estymacja opóźnienia (Theiler window) za pomocą wzajemnej informacji.
        theiler = estimate_delay(dataOJit, "mi_min")

        # PECUZAL draws its subsample from the task's default RNG: seed it from ours.
        samplesize < 1 && Random.seed!(rand(rng, UInt64))
        # Rekonstrukcja przestrzeni fazowej algorytmem PECUZAL.
        Y, tau_vals, ts_vals, Ls, epss = pecuzal_embedding(dataOJit; τs=0:msize, w=theiler, econ=true,
                                                           verbose=false, samplesize=samplesize)
        # With no embedding gain PECUZAL returns the raw series (tau_vals=[0]); keep Y m-dimensional.
        Y = StateSpaceSet(Y)
    else
        # Delays chosen earlier (from older data): only the delay vectors of this window are new.
        tau_vals, theiler = embedding.tau_vals, embedding.theiler
        Y = length(tau_vals) == 1 ? StateSpaceSet(dataOJit) : genembed(dataOJit, tau_vals)
        ts_vals, Ls, epss = ones(Int, length(tau_vals)), Float64[], nothing
    end

    # Wyodrębnienie rzeczywistych punktów historycznych do późniejszej walidacji.
    # Y[n] = (s[n+τ₁], …, s[n+τₘ]), so coordinate j of the next point Y[end+1]
    # is s[N+1-(τmax-τⱼ)]; the τmax coordinate is the future value s[N+1].
    tmax = maximum(tau_vals)
    realPoint = [s[end-backSize+1-(tmax-t)] for t in tau_vals]

    return (Y=Y, tau_vals=tau_vals, ts_vals=ts_vals, Ls=Ls, epss=epss, theiler=theiler, dataOJit=dataOJit, realPoint=realPoint)
end

"""
    EmbeddingBlocks(s; every, windowSize, msize, seed, samplesize)

Chooses the embedding (PECUZAL delays and the Theiler window) once per block of `every`
consecutive offsets instead of once per point. Offset b belongs to the block whose anchor is
the largest offset in it, cld(b, every)·every; the embedding is chosen on the window that ends
at that anchor, i.e. before every point of the block, so no point sees its own future.
Blocks are computed on demand, once, and shared between threads.
"""
struct EmbeddingBlocks
    s::Vector{Float64}
    every::Int
    windowSize::Int
    msize::Int
    seed::Int
    samplesize::Float64
    tasks::Dict{Int,Task}
    lock::ReentrantLock
end

EmbeddingBlocks(s; every, windowSize=82000, msize=10, seed=1, samplesize=1.0) =
    EmbeddingBlocks(Vector{Float64}(s), every, windowSize, msize, seed, samplesize, Dict{Int,Task}(), ReentrantLock())

anchor_offset(eb::EmbeddingBlocks, b) = cld(b, eb.every) * eb.every

function embedding_for(eb::EmbeddingBlocks, b)
    a = anchor_offset(eb, b)
    t = lock(eb.lock) do
        get!(eb.tasks, a) do
            Threads.@spawn begin
                r = run_series(eb.s, a; windowSize=eb.windowSize, msize=eb.msize, samplesize=eb.samplesize,
                               rng=Xoshiro(hash((eb.seed, :anchor, a))))
                (tau_vals=r.tau_vals, theiler=r.theiler, anchor=a)
            end
        end
    end
    return fetch(t)
end

const DEFAULT_DATAFILE = joinpath(@__DIR__, "..", "data", "btc_cryptocompare.csv")

# Loads a CryptoCompare-style CSV and adds the deltaO/deltaC columns.
function load_deltas(datafile=DEFAULT_DATAFILE; rng::AbstractRNG=Random.default_rng())
    mydf = NumberCruncher.DataManager.LoadData(datafile)
    NumberCruncher.DataManager.AddDeltas!(mydf; rng=rng)
    return mydf
end

# The original BTC pipeline: hourly open-price deltas from a CryptoCompare CSV.
function reconstruct(backSize; windowSize=82000, msize=10, datafile=DEFAULT_DATAFILE,
                     rng::AbstractRNG=Random.default_rng())
    mydf = load_deltas(datafile; rng=rng)
    r = run_series(mydf.deltaO, backSize; windowSize=windowSize, msize=msize, rng=rng)
    return merge(r, (DTFrame=mydf,))
end

# Old name, kept for existing scripts (`NumberCruncher.run(...)`).
run(args...; kwargs...) = reconstruct(args...; kwargs...)

# ==============================================================================
# NEIGHBOURHOOD ANALYSIS & PREDICTION
# ==============================================================================

# Estimates the next point of the m-dimensional reconstruction from k neighbours.
analyzeit2(Y, df::DataFrame, taus, backSize; kwargs...) = analyzeit2(Y, df.deltaO, taus, backSize; kwargs...)

# s is the prepared series the reconstruction Y was built from.
# theiler: states closer than this in time to Y[end] lie on the same stretch of trajectory
# and are not real neighbours; pass the w from run_series. The default 1 keeps the old Y[1:end-2].
# shadow=false: plain k-NN analogues, weighted by the embedding distance only.
# tree: optional KDTree of Y[1:end-1-theiler], shared between calls for different k.
function analyzeit2(Y, s::AbstractVector, taus, backSize; k=Analysis.DEFAULT_K, theiler::Integer=1,
                    shadow::Bool=true, tree=nothing)
    nsearch = length(Y) - 1 - theiler
    nsearch >= 1 || throw(ArgumentError("Theiler window $theiler leaves no candidate neighbours"))
    k = min(k, nsearch)
    neighs = tree === nothing ? Analysis.find_neighbours(Y[begin:nsearch], Y[end], k) :
                                Analysis.find_neighbours(nothing, Y[end], k; tree=tree)
    tLen = length(taus)
    
    # Przygotowanie wektora referencyjnego z uwzględnieniem brakujących współrzędnych (nothing).
    # Y[n] = (s[n+τ₁], …, s[n+τₘ]) (DelayEmbeddings convention), so in Y[end+1]
    # coordinate j is s[N+1-(τmax-τⱼ)]. The unknown future value s[N+1] sits at
    # the coordinate with the largest delay, not at τ=0; PECUZAL delays are unsorted.
    rPoint = Vector{Union{Float64,Nothing}}(fill(nothing, tLen))
    tmax = maximum(taus)
    idx = argmax(taus)
    for j in 1:tLen
        if j != idx
            rPoint[j] = s[end-backSize+1-(tmax-taus[j])]
        end
    end
    
    # Obliczenie ważonego środka ciężkości przesuniętych sąsiadów.
    successors = Y[neighs.idxs .+ 1]
    weights = Analysis.neighbour_weights(successors, neighs.dists, rPoint; shadow=shadow)
    center = [Analysis.weighted_coordinate(successors, weights, j) for j in 1:tLen]
    return (center=center, neigs=neighs, searchedIndex=idx, rPoint=rPoint, successors=successors, weights=weights)
end

"""
Walk-forward estimate of the hidden value p.z[end-b+1] from everything before it.

One reconstruction of p.z serves every variant: for each k it gives the estimate of the
value and of its absolute value, with shadow weighting (known coordinates) and without.
Estimates are in the z scale; multiply by `scale` for the original scale (scale is 1 when
the series is not normalised). The row also carries the truth and the naive baselines.
"""
#
# blocks: an EmbeddingBlocks of p.z to reuse block embeddings (nothing = PECUZAL for every point).
function estimate_point(p::Preprocessor.PreparedSeries, b; ks=(Analysis.DEFAULT_K,), msize=10,
                        windowSize=82000, seed=1, samplesize=1.0, blocks=nothing)
    z = p.z
    n = length(z)
    t = n - b + 1
    emb = blocks === nothing ? nothing : embedding_for(blocks, b)
    r0 = run_series(z, b; windowSize=windowSize, msize=msize, rng=Xoshiro(hash((seed, b))),
                    samplesize=samplesize, embedding=emb)
    win = (n - b - length(r0.dataOJit) + 1):(n - b)
    est = Dict{String,Any}()
    idx = argmax(r0.tau_vals)
    # one tree for all k (building it per k cost more than the queries)
    tree = KDTree(r0.Y[begin:max(1, length(r0.Y) - 1 - r0.theiler)])
    for k in ks
        a = analyzeit2(r0.Y, z, r0.tau_vals, b; k=k, theiler=r0.theiler, tree=tree)
        w0 = Analysis.neighbour_weights(a.successors, a.neigs.dists, a.rPoint; shadow=false)
        est[string(k)] = (ret=a.center[idx], abs=Analysis.weighted_coordinate(a.successors, a.weights, idx; f=abs),
                          ret_ns=Analysis.weighted_coordinate(a.successors, w0, idx),
                          abs_ns=Analysis.weighted_coordinate(a.successors, w0, idx; f=abs),
                          center=a.center)
    end
    rw, zw = p.r[win], z[win]
    P = p.period
    return (offset=b, label=string(p.labels[t]), m=length(r0.tau_vals), taus=r0.tau_vals,
            anchor=emb === nothing ? b : emb.anchor,
            theiler=r0.theiler, searched_index=idx, real_point=r0.realPoint,
            real_r=p.r[t], last_r=p.r[t-1], real_z=z[t], last_z=z[t-1], scale=p.scale[t], ewma=p.ewma[t],
            win_mean_r=mean(rw), win_mean_abs_r=mean(abs, rw), win_mean_z=mean(zw), win_mean_abs_z=mean(abs, zw),
            ar1_r=Analysis.ar1_forecast(rw, p.r[t-1]), ar1_z=Analysis.ar1_forecast(zw, z[t-1]),
            # seasonal persistence: the value one period earlier (same time yesterday for hourly data)
            seasonal_r=P > 0 && t > P ? p.r[t-P] : NaN, seasonal_z=P > 0 && t > P ? z[t-P] : NaN,
            est=est)
end

# ==============================================================================
# THE GLOBAL SIMULATION (GRID SEARCH)
# ==============================================================================
function do_the_thing(backRange=1000:-20:10, neighRange=2:5:1000; seed=1, datafile=DEFAULT_DATAFILE)
    # Tasks can migrate between threads, so collect under a lock, not per threadid().
    results_all = Vector{NamedTuple}()
    results_lock = ReentrantLock()
    mydf = load_deltas(datafile)

    # Wielowątkowa pętla po zakresie testowym czasu.
    # :greedy hands out points one by one: their cost varies several-fold with the embedding.
    Threads.@threads :greedy for ix in 1:length(backRange)
        x = backRange[ix]
        local_results = Vector{NamedTuple}() 
 
        # Przeszukiwanie różnych parametrów osadzania (msize).
        for tausize in 5:10:100
            r0 = run_series(mydf.deltaO, x; msize=tausize, rng=Xoshiro(hash((seed, x, tausize))))
            println(r0.tau_vals)

            # Testowanie czułości modelu na liczbę sąsiadów (k).
            for y in neighRange
                # The tree holds Y[1:end-1-theiler].
                y_actual = min(y, length(r0.Y) - 1 - r0.theiler)

                # Wykonanie prognozy.
                r1 = analyzeit2(r0.Y, mydf.deltaO, r0.tau_vals, x; k=y_actual, theiler=r0.theiler)

                # est_x/real_x is the unknown (largest-delay) coordinate; est/real keep all m.
                push!(local_results, (
                    m_size=tausize,
                    time = x,
                    neighs = y_actual,
                    taus = r0.tau_vals,
                    searched_index = r1.searchedIndex,
                    est_x = r1.center[r1.searchedIndex],
                    real_x = r0.realPoint[r1.searchedIndex],
                    est = r1.center,
                    real = r0.realPoint,
                ))
            end
        end
        lock(results_lock) do
            append!(results_all, local_results)
        end
    end

    df = DataFrame(results_all)
    df.dists = abs.(df.est_x .- df.real_x)
    return df
end


# ==============================================================================
# WALK-FORWARD VALIDATION (THE THING 2)
# ==============================================================================
# Sequentially analyses the last N points to evaluate predictive performance.
# Each point gets its own seeded jitter, so a run is reproducible for a given seed.
#
# The series is `transform` (:diff or :logdiff) of `column`; aggregate=:daily (or :weekly)
# uses the daily `stat` (e.g. mean) first, and n_points then counts days (weeks).
# normalize=:ewma/:rolling embeds volatility-normalised values z = r/σ (σ ex-ante) and converts
# estimates back with σ; all scores stay in the original scale (est_z/real_z add the z scale).
# target=:abs_return estimates |r| instead of r; shadow=false uses plain k-NN weights.
# period (:auto = samples per day from the timestamps, e.g. 24 for hourly data) adds the seasonal
# persistence baseline "same time one period earlier"; transform=:sdiff removes that cycle first.
# reembed_every=B runs PECUZAL once per block of B points (on data before the block) instead of
# for every point; samplesize < 1 makes PECUZAL subsample (both much faster, both change results).
function do_the_thing2(n_points=100, tausize=30, k_neighs=50; seed=1, datafile=DEFAULT_DATAFILE,
                       windowSize=82000, column=:open, transform::Symbol=:diff,
                       aggregate::Symbol=:none, stat::Symbol=:mean,
                       normalize::Symbol=:none, ewma_lambda=0.98, vol_window=168, vol_burnin=500,
                       target::Symbol=:return, shadow::Bool=true, period=:auto,
                       reembed_every::Integer=1, samplesize::Real=1.0)
    target in (:return, :abs_return) || throw(ArgumentError("target must be :return or :abs_return"))
    # Auto-detection only for sub-daily data; after daily resampling pass e.g. period=7 explicitly.
    autoperiod(l) = period == :auto ? (aggregate == :none ? Preprocessor.detect_period(l) : 0) : Int(period)
    if aggregate == :none && column == :open && transform == :diff
        mydf = load_deltas(datafile)            # the original hourly deltaO series
        s, labels = mydf.deltaO, DataManager.RowLabels(mydf)
        P = autoperiod(labels)
    else
        raw = DataManager.LoadData(datafile)
        rawlabels = DataManager.RowLabels(raw)
        P = autoperiod(rawlabels)
        prep = Preprocessor.prepare_series(raw[!, column], rawlabels; aggregate=aggregate, stat=stat,
                                           transform=transform, period=P)
        s, labels = prep.series, prep.labels
    end
    p = Preprocessor.normalize_series(s, labels; normalize=normalize, ewma_lambda=ewma_lambda,
                                      vol_window=vol_window, vol_burnin=vol_burnin, period=P)
    n_points <= length(p.z) ÷ 2 || throw(ArgumentError("only $(length(p.z)) values; test at most $(length(p.z) ÷ 2) points"))
    blocks = reembed_every > 1 ? EmbeddingBlocks(p.z; every=reembed_every, windowSize=windowSize, msize=tausize,
                                                 seed=seed, samplesize=samplesize) : nothing
    backRange = n_points:-1:1

    # Thread-safe container
    results_all_vec = Vector{NamedTuple}()
    results_lock = ReentrantLock()

    println("Commencing walk-forward analysis for $n_points points using $(nthreads()) threads...")

    # :greedy hands out points one by one: their cost varies several-fold with the embedding.
    Threads.@threads :greedy for ix in 1:length(backRange)
        x = backRange[ix]
        println("Processing point: $x")
        e = estimate_point(p, x; ks=(k_neighs,), msize=tausize, windowSize=windowSize, seed=seed,
                           samplesize=samplesize, blocks=blocks)
        v = e.est[string(k_neighs)]
        ez = target == :return ? (shadow ? v.ret : v.ret_ns) : (shadow ? v.abs : v.abs_ns)
        f = target == :return ? identity : abs
        println("iAfter analysis, offset: $x, idx $(e.searched_index) tau vals: $(e.taus) est: $(v.center) real: $(e.real_point)")
        lock(results_lock) do
            push!(results_all_vec, (
                time_offset = x,
                taus = e.taus,
                theiler = e.theiler,
                searched_index = e.searched_index,
                est_x = ez * e.scale,               # original scale
                real_x = f(e.real_r),
                persistence_x = f(e.last_r),        # baseline: the last known value
                ar1_x = e.ar1_r,
                ewma_x = e.ewma * sqrt(2 / π),      # baseline for |r|: expected |r| under N(0, σ²)
                seasonal_x = f(e.seasonal_r),       # baseline: same time one period earlier
                est_z = ez,
                real_z = f(e.real_z),
                sigma = e.scale,
                est = v.center,
                real = e.real_point,
            ))
        end
    end

    df = DataFrame(results_all_vec)
    sort!(df, :time_offset, rev=true)
    df.dists = abs.(df.est_x .- df.real_x)
    metadata!(df, "target", string(target))
    metadata!(df, "normalize", Preprocessor.describe_normalization(p))
    return df
end

# Skill of a do_the_thing2 result against the naive predictors (zero, persistence and AR(1)
# for returns; EWMA and persistence for |r|).
function summarize(df::DataFrame)
    seasonal = "seasonal_x" in names(df) && all(isfinite, df.seasonal_x) ? df.seasonal_x : nothing
    if metadata(df, "target", "return") == "abs_return"
        return Analysis.abs_skill_summary(df.est_x, df.real_x, df.persistence_x, df.ewma_x; seasonal=seasonal)
    end
    return Analysis.skill_summary(df.est_x, df.real_x, df.persistence_x;
                                  ar1="ar1_x" in names(df) ? df.ar1_x : nothing, seasonal=seasonal)
end



# ==============================================================================
# VISUALIZATION
# ==============================================================================
using PlotlyJS

function plotit(r)
    plt = Plot(
        scatter3d(
            x = r.m_size,
            y = r.neighs,
            z = r.dists,
            mode = "markers",
            marker = attr(
                size = 2,
                color = r.dists,
                colorscale = "Viridis",
                showscale = true
            ),
            name = "Distance points"
        ),
        Layout(
            title = "3D Hyperparameter Optimization",
            scene = attr(
                xaxis = attr(title="Embedding Size"),
                yaxis = attr(title="Neighbours (k)"),
                zaxis = attr(title="Euclidean Distance")
            )
        )
    )
    display(plt)
end


# Detailed error analysis plot.
function plot_error_analysis(df)
    error_x = df.est_x .- df.real_x
    
    trace1 = PlotlyJS.scatter(y=df.dists, name="Total Euclidean Error", 
                     mode="lines", fill="tozeroy", line=PlotlyJS.attr(color="rgba(100, 100, 250, 0.5)"))
    
    trace2 = PlotlyJS.scatter(y=error_x, name="Residual X (Est - Real)", 
                     mode="lines+markers", marker=PlotlyJS.attr(size=5, color="firebrick"))
    
    trace3 = PlotlyJS.scatter(y=zeros(nrow(df)), name="Zero Error Baseline", 
                     mode="lines", line=PlotlyJS.attr(color="black", dash="dot", width=1))
    
    p = PlotlyJS.Plot([trace1, trace2, trace3], 
        PlotlyJS.Layout(
            title="Comprehensive Prediction Error Analysis",
            xaxis=PlotlyJS.attr(title="Chronological Step"),
            yaxis=PlotlyJS.attr(title="Error Value"),
            legend=PlotlyJS.attr(orientation="h", y=-0.2),
            hovermode="x unified"
        )
    )
    display(p)
end

# ==============================================================================
# WIZUALIZACJA I METRYKI
# ==============================================================================

function plot_aesthetic_analysis(df)
    # Główne zmienne (Interesuje nas X - czyli nasza prognozowana delta)
    error_x = df.est_x .- df.real_x
    n_rows = size(df, 1)
    
    # Statystyki
    mae_x = round(sum(abs.(error_x)) / n_rows, digits=4)
    same_direction = (sign.(df.est_x) .== sign.(df.real_x))
    hit_rate = round(sum(same_direction) / n_rows * 100, digits=2)

    # 1. Porównanie Real vs Est (X)
    t1 = PlotlyJS.scatter(y=df.real_x, name="Rate Change", mode="lines+markers", 
                          line=PlotlyJS.attr(color="#2c3e50", width=2), 
                          marker=PlotlyJS.attr(size=5), xaxis="x1", yaxis="y1")
    t2 = PlotlyJS.scatter(y=df.est_x, name="Estimate", mode="lines+markers", 
                          line=PlotlyJS.attr(color="#e67e22", width=2, dash="dot"), 
                          marker=PlotlyJS.attr(size=5, symbol="diamond"), xaxis="x1", yaxis="y1")

    # 2. Histogram błędów (Czy błąd jest symetryczny?)
    t3 = PlotlyJS.histogram(x=error_x, name="Error distribution", 
                            marker=PlotlyJS.attr(color="#3498db"),
                            nbinsx=5, xaxis="x2", yaxis="y2")

    # 3. Residua w czasie
    t4 = PlotlyJS.scatter(y=error_x, name="Points Errors", mode="markers",
                          marker=PlotlyJS.attr(color=ifelse.(error_x .> 0, "#27ae60", "#c0392b"), size=8),
                          xaxis="x3", yaxis="y3")
    t5 = PlotlyJS.scatter(y=zeros(n_rows), name="Perfect estimate", mode="lines", 
                          line=PlotlyJS.attr(color="black", width=1), xaxis="x3", yaxis="y3", showlegend=false)

    layout = PlotlyJS.Layout(
        title=PlotlyJS.attr(text="<b>Analiza Predykcji: Skupienie na wymiarze X</b><br><sup>Hit Rate: $hit_rate% | MAE: $mae_x</sup>", x=0.5),
        grid=PlotlyJS.attr(rows=3, columns=1, pattern="independent"),
        template="plotly_white",
        height=1000,
        hovermode="x unified",
        legend=PlotlyJS.attr(orientation="h", x=0.5, xanchor="center", y=-0.05),
        
        xaxis1=PlotlyJS.attr(title="Obserwacje"),
        yaxis1=PlotlyJS.attr(title="Wartość Delta"),
        
        xaxis2=PlotlyJS.attr(title="Wartość błędu"),
        yaxis2=PlotlyJS.attr(title="Częstotliwość"),
        
        xaxis3=PlotlyJS.attr(title="Obserwacje"),
        yaxis3=PlotlyJS.attr(title="Różnica (Est-Real)")
    )

    p = PlotlyJS.Plot([t1, t2, t3, t4, t5], layout)
    display(p)
end



# ==============================================================================
# BROWSER GUI
# ==============================================================================
include("GUI/GUI.jl")
using .GUI: serve_gui

end # Closes module NumberCruncher
