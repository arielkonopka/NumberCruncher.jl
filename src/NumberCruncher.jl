module NumberCruncher
using NearestNeighbors
include("DataManager/DataManager.jl")
using .DataManager
include("Analysis/Analysis.jl")
using .Analysis
using DynamicalSystems
using DataFrames
using Base.Threads
using Makie
using HTTP
using JSON
using CSV
using Dates

export run, do_the_thing2, plot_aesthetic_analysis, update_bitcoin_data!
const CRYPTOCOMPARE_KEY = strip(read(
    joinpath(@__DIR__, "..", "config", "cryptocompare_key.txt"),
    String
))

# ==============================================================================
# Data Fetch (CRYPTOCOMPARE - PORT Z PYTHONA)
# ==============================================================================
"""
Downloades historical data from z CryptoCompare, from start_date to current time.

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
    
    try
        while current_end_ts > start_ts
            url = "https://min-api.cryptocompare.com/data/v2/histohour"
            params = [
                "fsym" => symbol,
                "tsym" => "USD",
                "limit" => string(limit),
                "toTs" => string(current_end_ts),
                "api_key" => CRYPTOCOMPARE_KEY
            ]
            
            # Wysłanie zapytania
            query_str = join(["$k=$v" for (k,v) in params], "&")
            response = HTTP.get(url * "?" * query_str)
            data = JSON.parse(String(response.body))
            
            if data["Response"] != "Success"
                println("Błąd API: $(data["Message"])")
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
        end
        
    catch e
        println("Wystąpił błąd: $e")
    end
end








# ==============================================================================
# PHASE SPACE RECONSTRUCTION (PECUZAL)
# ==============================================================================
# Funkcja przygotowuje manifest (przestrzeń fazową) dla konkretnego punktu w czasie.
# 'backSize' określa przesunięcie od końca danych w celu przeprowadzenia testu.
function run(backSize; windowSize=82000, msize=10)
    # Pobieranie danych i dodawanie jittera (szumu zapobiegającego błędom k-NN).
    mydf = NumberCruncher.DataManager.LoadData("./data/btc_cryptocompare.csv")
    NumberCruncher.DataManager.AddDeltas!(mydf)
    
    # Określenie okna czasowego obserwacji.
    windowStart = floor(max(1, length(mydf.deltaOJittered) - backSize - windowSize))
    dataOJit = mydf.deltaOJittered[windowStart:end-backSize]

    # Estymacja opóźnienia (Theiler window) za pomocą wzajemnej informacji.
    theiler = estimate_delay(dataOJit, "mi_min")

    # Rekonstrukcja przestrzeni fazowej algorytmem PECUZAL.
    Y, tau_vals, ts_vals, Ls, epss = pecuzal_embedding(dataOJit; τs=0:msize, w=theiler, econ=true, verbose=false)

    # Wyodrębnienie rzeczywistych punktów historycznych do późniejszej walidacji.
    realPoint = fill(0.0, length(tau_vals))
    c = 1
    for i in reverse(tau_vals)
        realPoint[c] = mydf.deltaO[end-backSize-i+1]
        c += 1
    end
    
    return (Y=Y, tau_vals=tau_vals, ts_vals=ts_vals, Ls=Ls, epss=epss, DTFrame=mydf, dataOJit=dataOJit, realPoint=realPoint)
end

# ==============================================================================
# NEIGHBOURHOOD ANALYSIS & PREDICTION
# ==============================================================================

# Wersja 1: Estymacja na podstawie jednego znanego wymiaru (X).
function analyzeit(Y, df, taus, backSize)
    neighs = NumberCruncher.Analysis.find_neighbours(Y[begin:end-2], Y[end])
    tau = maximum(taus)
    center = NumberCruncher.Analysis.weighted_center(
        Y[neighs.idxs .+ 1],
        neighs.dists,
        df.deltaO[end-backSize-tau+1],
        nothing
    )
    return (center=center, neigs=neighs)
end

# Wersja 2: Bardziej zaawansowana estymacja wektorowa (używana w głównej pętli).
function analyzeit2(Y, df, taus, backSize)
    neighs = NumberCruncher.Analysis.find_neighbours(Y[begin:end-2], Y[end])
    tLen = length(taus)
    
    # Przygotowanie wektora referencyjnego z uwzględnieniem brakujących współrzędnych (nothing).
    rPoint = Vector{Union{Float64,Nothing}}(fill(nothing, tLen))
    reversed_taus = reverse(taus)
    idx=1
    for i in 1:tLen
        if(reversed_taus[i] != 0)
            rPoint[i] = df.deltaO[end-backSize+1-reversed_taus[i]]
        else
            rPoint[i] = nothing
            idx=i
        end
    end
    
    # Obliczenie ważonego środka ciężkości przesuniętych sąsiadów.
    center = NumberCruncher.Analysis.weighted_center2(
        Y[neighs.idxs .+ 1],
        neighs.dists,
        rPoint
    )
    return (center=center, neigs=neighs,searchedIndex=idx)
end

# ==============================================================================
# THE GLOBAL SIMULATION (GRID SEARCH)
# ==============================================================================
function do_the_thing(backRange=1000:-20:10, neighRange=2:5:1000)
    nthreads = Threads.nthreads()
    # Buforowanie wyników na poziomie wątków dla uniknięcia "race conditions".
    results_threads = [Vector{NamedTuple}() for _ in 1:nthreads]

    # Wielowątkowa pętla po zakresie testowym czasu.
    Threads.@threads for ix in 1:length(backRange)
        x = backRange[ix]
        local_results = Vector{NamedTuple}() 
 
        # Przeszukiwanie różnych parametrów osadzania (msize).
        for tausize in 5:10:100
            r0 = run(x, msize=tausize)
            println(r0.tau_vals)

            # Testowanie czułości modelu na liczbę sąsiadów (k).
            for y in neighRange
                y_actual = min(y, length(r0.Y)-1)
                Analysis.neighs = y_actual 

                # Wykonanie prognozy.
                r1 = analyzeit2(r0.Y, r0.DTFrame, r0.tau_vals, x)

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
        append!(results_threads[threadid()], local_results)
    end
    
    # Agregacja i obliczenie błędu euklidesowego.
    results_all = vcat(results_threads...)
    df = DataFrame(results_all)
    df.dists = sqrt.((df.est_x .- df.real_x).^2 .+ (df.est_y .- df.real_y).^2)
    return df
end


# ==============================================================================
# WALK-FORWARD VALIDATION (THE THING 2)
# ==============================================================================
# Sequentially analyses the last N points to evaluate predictive performance.
function do_the_thing2(n_points=100, tausize=30, k_neighs=50)
    backRange = n_points:-1:1
    
    # Thread-safe container
    results_all_vec = Vector{NamedTuple}()
    results_lock = ReentrantLock()

    println("Commencing walk-forward analysis for $n_points points using $(nthreads()) threads...")

    Threads.@threads for ix in 1:length(backRange)
        x = backRange[ix]
        
        # Logging progress
        println("Processing point: $x")
        
        # Use a local copy of parameters or set them safely
        Analysis.neighs = k_neighs 
        r0 = run(x)
        
        println("Executing state estimation for offset: $x")
        r1 = analyzeit2(r0.Y, r0.DTFrame, r0.tau_vals, x)
        #idx_now = argmax(r0.tau_vals)
        println("iAfter analysis, offset: $x, idx $(r1.searchedIndex) tau vals: $(r0.tau_vals) est: $(r1.center) real: $(r0.realPoint)")
        lock(results_lock) do
            push!(results_all_vec, (
                time_offset = x,
                est_x = r1.center[r1.searchedIndex],
                real_x = r0.realPoint[r1.searchedIndex],
            ))
        end
    end

    df = DataFrame(results_all_vec)
    sort!(df, :time_offset, rev=true)
    df.dists = sqrt.((df.est_x .- df.real_x).^2)
    return df
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



end # Closes module NumberCruncher
