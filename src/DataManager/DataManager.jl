module DataManager
using  CSV
using DataFrames
using Dates
using DynamicalSystems
using HTTP
using JSON
using Random

"""
Loads data from a CSV file into a DataFrame

Parameters:
- filename::AbstractString: path to the CSV file

Returns:
- DataFrame: loaded data
"""
function LoadData(filename::AbstractString)
    rawFile = CSV.File(filename)
    return DataFrame(rawFile)
end

"""
Adds delta columns (price changes) to the DataFrame

Parameters:
- df::DataFrame: input DataFrame with 'open' and 'close' columns

Returns:
- DataFrame: modified DataFrame with added 'deltaO' and 'deltaC' and 'deltaLogC' and 'deltaLogO' columns
the 'log' columns are subtractions of logarithms od respectable Open or Close columns
we also do the cleaning up of the missing values
"""
function AddDeltas!(df::DataFrame; rng::AbstractRNG=Random.default_rng())
    # select!(df, Not([:conversionSymbol, :conversionType]))
    df.deltaO = vcat([df.open[i+1] - df.open[i] for i in 1:(nrow(df)-1)], missing)
    df.deltaC = vcat([df.close[i+1] - df.close[i] for i in 1:(nrow(df)-1)], missing)
   # df.deltaLogO=vcat([log(df.open[i+1]) - log(df.open[i]) for i in 1:(nrow(df)-1)], missing)
   # df.deltaLogC=vcat([log(df.close[i+1]) - log(df.close[i]) for i in 1:(nrow(df)-1)], missing)
    # Only the deltas matter; mostly-empty columns like conversionSymbol must not drop rows.
    dropmissing!(df, [:deltaO, :deltaC], disallowmissing=true)

    df.deltaCJittered=df.deltaC.+1e-10.*randn(rng, length(df.deltaC))
    df.deltaOJittered=df.deltaO.+1e-10.*randn(rng, length(df.deltaO))

end

"""
Generates a Lorenz attractor trajectory

Parameters:
- N: number of samples kept, Δt: sampling step, Ttr: transient time thrown away
- σ, ρ, β: Lorenz parameters, u0: initial condition

Returns:
- DataFrame with columns t, x, y, z
"""
function GenerateLorenz(; N::Integer=20000, Δt::Real=0.01, Ttr::Real=100.0,
                        σ::Real=10.0, ρ::Real=28.0, β::Real=8/3, u0=[0.0, 10.0, 0.0])
    ds = PredefinedDynamicalSystems.lorenz(Float64.(u0); σ=σ, ρ=ρ, β=β)
    X, t = trajectory(ds, (N - 1) * Δt; Δt=Δt, Ttr=Ttr)
    return DataFrame(t=collect(t) .- Ttr, x=X[:, 1], y=X[:, 2], z=X[:, 3])
end

"""
Names of the columns that hold numbers (candidates for the analysed series)
"""
NumericColumns(df::DataFrame) =
    [n for n in names(df) if nonmissingtype(eltype(df[!, n])) <: Real]

"""
One human-readable label per row: a 'timestamp' (ms), date/time or 't' column, else the row number
"""
function RowLabels(df::DataFrame)
    lower = Dict(lowercase(n) => n for n in names(df))
    if haskey(lower, "timestamp") && nonmissingtype(eltype(df[!, lower["timestamp"]])) <: Real
        return [ismissing(x) ? "" : string(unix2datetime(x / 1000)) for x in df[!, lower["timestamp"]]]
    end
    for c in ("time", "date", "datetime", "observation_date", "timestamp", "t")
        haskey(lower, c) && return [ismissing(x) ? "" : string(x) for x in df[!, lower[c]]]
    end
    return string.(1:nrow(df))
end

# ==============================================================================
# IMPORTERS (no API key needed)
# ==============================================================================
const YAHOO_SYMBOL = r"^[A-Za-z0-9.^=_-]{1,20}$"
const YAHOO_INTERVALS = ("1d", "1wk", "1mo", "1h")
const FRED_ID = r"^[A-Za-z0-9_]{1,40}$"
const METEO_VARIABLES = ("temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl",
                         "surface_pressure", "precipitation", "wind_speed_10m", "wind_direction_10m",
                         "cloud_cover", "shortwave_radiation")

tofloat(x) = x === nothing ? missing : Float64(x)

"""
Daily/weekly/monthly (or hourly, last 730 days) prices from Yahoo Finance's chart API.
Unofficial endpoint: it may change or rate-limit without notice.

Returns:
- DataFrame with columns time (UTC), open, high, low, close, adjclose, volume
"""
function FetchYahoo(symbol::AbstractString; interval::AbstractString="1d")
    occursin(YAHOO_SYMBOL, symbol) || throw(ArgumentError("invalid symbol '$symbol'"))
    interval in YAHOO_INTERVALS || throw(ArgumentError("interval must be one of $YAHOO_INTERVALS"))
    # An explicit period: range=max silently coarsens daily data to ~quarterly.
    now_ts = round(Int, datetime2unix(now(UTC)))
    from_ts = interval == "1h" ? now_ts - 729 * 86400 : -2208988800   # 1h: last 730 days; else from 1900
    url = "https://query1.finance.yahoo.com/v8/finance/chart/$(HTTP.escapeuri(symbol))?period1=$from_ts&period2=$now_ts&interval=$interval&events=div%2Csplit"
    resp = HTTP.get(url, ["User-Agent" => "Mozilla/5.0 (NumberCruncher.jl)"]; status_exception=false)
    data = JSON.parse(String(resp.body))
    chart = get(data, "chart", Dict())
    err = get(chart, "error", nothing)
    err !== nothing && throw(ArgumentError("Yahoo: $(get(err, "description", err))"))
    resp.status == 200 || throw(ErrorException("Yahoo returned HTTP $(resp.status)"))
    r = chart["result"][1]
    haskey(r, "timestamp") || throw(ArgumentError("Yahoo returned no prices for '$symbol'"))
    q = r["indicators"]["quote"][1]
    adj = haskey(r["indicators"], "adjclose") ? r["indicators"]["adjclose"][1]["adjclose"] : q["close"]
    return DataFrame(time=[unix2datetime(t) for t in r["timestamp"]],
                     open=tofloat.(q["open"]), high=tofloat.(q["high"]), low=tofloat.(q["low"]),
                     close=tofloat.(q["close"]), adjclose=tofloat.(adj), volume=tofloat.(q["volume"]))
end

"""
A FRED series (e.g. SP500, DJIA, NASDAQCOM, VIXCLS) as published by the St. Louis Fed.

Returns:
- DataFrame with columns observation_date and the series id
"""
function FetchFRED(series_id::AbstractString)
    occursin(FRED_ID, series_id) || throw(ArgumentError("invalid FRED series id '$series_id'"))
    resp = HTTP.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=$series_id"; status_exception=false)
    resp.status == 200 || throw(ArgumentError("FRED returned HTTP $(resp.status) for '$series_id'"))
    df = CSV.read(resp.body, DataFrame; missingstring=["", "."])
    ncol(df) >= 2 || throw(ArgumentError("FRED returned no data for '$series_id'"))
    return df
end

"""
Hourly historical weather (ERA5 reanalysis) from the Open-Meteo archive API.

Parameters:
- latitude, longitude: location; start_date, end_date: Date range (from 1940)
- variables: subset of METEO_VARIABLES

Returns:
- DataFrame with column time (UTC) and one column per variable
"""
function FetchOpenMeteo(latitude::Real, longitude::Real, start_date::Date, end_date::Date;
                        variables=["temperature_2m"])
    -90 <= latitude <= 90 && -180 <= longitude <= 180 || throw(ArgumentError("invalid coordinates"))
    start_date <= end_date || throw(ArgumentError("start date is after end date"))
    bad = setdiff(variables, METEO_VARIABLES)
    isempty(bad) && !isempty(variables) || throw(ArgumentError("unknown weather variables: $bad"))
    url = "https://archive-api.open-meteo.com/v1/archive?latitude=$latitude&longitude=$longitude" *
          "&start_date=$start_date&end_date=$end_date&hourly=$(join(variables, ","))&timezone=UTC"
    resp = HTTP.get(url; status_exception=false)
    data = JSON.parse(String(resp.body))
    resp.status == 200 || throw(ArgumentError("Open-Meteo: $(get(data, "reason", "HTTP $(resp.status)"))"))
    h = data["hourly"]
    df = DataFrame(time=string.(h["time"]))
    for v in variables
        df[!, v] = tofloat.(h[v])
    end
    return df
end

end