module Preprocessor
using Dates
using Statistics

export TRANSFORMS, AGGREGATES, STATS, NORMALIZATIONS, PreparedSeries, prepare_series, aggregate_series,
       normalize_series, ewma_volatility, rolling_volatility

"""
Available preparation steps:
- :raw     - the values as they are (e.g. a Lorenz coordinate)
- :diff    - first differences v[i+1]-v[i] (what deltaO is for prices)
- :logdiff - log returns log(v[i+1])-log(v[i]); needs positive values
"""
const TRANSFORMS = (:raw, :diff, :logdiff)

"""
Resampling applied before the transform: :none, :daily or :weekly (weeks start on Monday)
"""
const AGGREGATES = (:none, :daily, :weekly)

"""
Statistic of each daily/weekly bucket
"""
const STATS = (:mean, :median, :last, :min, :max)

"""
Parses a row label as a date/time; `nothing` for labels that are not dates (e.g. row numbers)
"""
function parse_time(x)
    x isa DateTime && return x
    x isa Date && return DateTime(x)
    str = string(x)
    occursin(r"^\d{4}-\d{2}-\d{2}", str) || return nothing
    return tryparse(DateTime, replace(str, " " => "T"))
end

"""
Resamples a series into daily or weekly buckets.

Parameters:
- values, labels: the series and its date/time labels (in any order)
- by: :daily or :weekly
- stat: one of STATS
- min_fraction: buckets with fewer values than this fraction of the median bucket size
  are dropped (e.g. today's partial day, or days with gaps)

Returns:
- (series, labels): one value per bucket in time order, labelled with the bucket's first day
"""
function aggregate_series(values::AbstractVector{<:Real}, labels::AbstractVector;
                          by::Symbol=:daily, stat::Symbol=:mean, min_fraction::Real=0.8)
    by in (:daily, :weekly) || throw(ArgumentError("unknown aggregation $by; use :daily or :weekly"))
    stat in STATS || throw(ArgumentError("unknown statistic $stat; use one of $STATS"))
    times = parse_time.(labels)
    any(isnothing, times) &&
        throw(ArgumentError("aggregation needs a date/time column, but a label like '$(labels[findfirst(isnothing, times)])' is not a date"))
    order = sortperm(times)
    key(t) = by == :daily ? Date(t) : firstdayofweek(Date(t))
    buckets = Date[]
    groups = Vector{Vector{Float64}}()
    for i in order
        k = key(times[i])
        if isempty(buckets) || buckets[end] != k
            push!(buckets, k)
            push!(groups, Float64[])
        end
        push!(groups[end], values[i])
    end
    f = stat == :mean ? mean : stat == :median ? median : stat == :last ? last : stat == :min ? minimum : maximum
    full = length.(groups) .>= min_fraction * median(length.(groups))
    return (series=[f(g) for g in groups[full]], labels=string.(buckets[full]))
end

# ==============================================================================
# VOLATILITY NORMALISATION
# ==============================================================================
"""
Volatility normalisations: :none, :ewma (RiskMetrics) or :rolling (past-window std)
"""
const NORMALIZATIONS = (:none, :ewma, :rolling)

"""
Ex-ante EWMA volatility: σ²[1] = var(r[1:init_n]), σ²[t] = λσ²[t-1] + (1-λ)r[t-1]².
σ[t] uses only r[1..t-1] (plus the init window, which the caller must burn in).
"""
function ewma_volatility(r::AbstractVector{<:Real}; λ::Real=0.98, init_n::Integer=500)
    0 < λ < 1 || throw(ArgumentError("λ must be in (0, 1)"))
    n = length(r)
    σ² = Vector{Float64}(undef, n)
    σ²[1] = var(@view r[1:min(init_n, n)])
    for t in 2:n
        σ²[t] = λ * σ²[t-1] + (1 - λ) * r[t-1]^2
    end
    return sqrt.(σ²)
end

"""
Ex-ante rolling volatility: σ[t] = std(r[t-W:t-1]), NaN for t ≤ W.
"""
function rolling_volatility(r::AbstractVector{<:Real}; W::Integer=168)
    W >= 2 || throw(ArgumentError("the volatility window must be at least 2"))
    σ = fill(NaN, length(r))
    for t in W+1:length(r)
        σ[t] = std(@view r[t-W:t-1])
    end
    return σ
end

"""
A prepared series together with its volatility scale.

- r: the series in its original scale (e.g. log returns)
- z: what gets embedded: r ./ scale (equal to r when normalize == :none)
- scale: σ used for normalisation, 1 everywhere when normalize == :none;
  scale[t] depends only on r[1..t-1], so an estimate ẑ[t] converts back as ẑ[t]·scale[t]
- ewma: EWMA σ (always computed), the baseline for |r| forecasts
"""
struct PreparedSeries
    r::Vector{Float64}
    z::Vector{Float64}
    scale::Vector{Float64}
    ewma::Vector{Float64}
    labels::Vector
    normalize::Symbol
    ewma_lambda::Float64
    vol_window::Int
    vol_burnin::Int
end

"""
Normalises r by an ex-ante volatility estimate and drops the burn-in.

With normalize = :none nothing is dropped and z === r, so results are identical to
working on r directly. Otherwise the first vol_burnin points are dropped (at least the
EWMA init window, or the rolling window), so the volatility estimate has settled.
"""
function normalize_series(r::AbstractVector{<:Real}, labels::AbstractVector=1:length(r);
                          normalize::Symbol=:none, ewma_lambda::Real=0.98, vol_window::Integer=168,
                          vol_burnin::Integer=500, init_n::Integer=500)
    normalize in NORMALIZATIONS || throw(ArgumentError("unknown normalisation $normalize; use one of $NORMALIZATIONS"))
    r = Vector{Float64}(r)
    labels = collect(labels)
    ewma = ewma_volatility(r; λ=ewma_lambda, init_n=init_n)
    if normalize == :none
        return PreparedSeries(r, r, ones(length(r)), ewma, labels, normalize, ewma_lambda, vol_window, 0)
    end
    σ = normalize == :ewma ? ewma : rolling_volatility(r; W=vol_window)
    burn = max(vol_burnin, normalize == :ewma ? init_n : vol_window)
    length(r) > burn + 10 || throw(ArgumentError("series too short for a burn-in of $burn points"))
    keep = burn+1:length(r)
    σk = σ[keep]
    floor_ = 1e-12 * max(median(σk), eps())
    σk = max.(σk, floor_)                   # a flat stretch must not give z = ±Inf
    return PreparedSeries(r[keep], r[keep] ./ σk, σk, ewma[keep], labels[keep], normalize,
                          ewma_lambda, vol_window, burn)
end

"""
Human-readable name of a normalisation, e.g. "EWMA-normalised (λ = 0.98)"
"""
function describe_normalization(p::PreparedSeries)
    p.normalize == :ewma && return "EWMA-normalised (λ = $(p.ewma_lambda))"
    p.normalize == :rolling && return "rolling-σ-normalised (W = $(p.vol_window))"
    return ""
end

"""
Turns a raw column into the series the estimator works on.

Parameters:
- values: raw column, may contain `missing`/NaN (those rows are skipped)
- labels: one label per raw row (time stamps, row numbers, ...)
- transform: one of TRANSFORMS
- aggregate, stat: optional daily/weekly resampling (see AGGREGATES, STATS), applied before
  the transform, e.g. aggregate=:daily, stat=:mean, transform=:diff gives changes of daily means
- noise: Gaussian noise added after the transform, as a fraction of the series' std

Returns:
- (series, labels): series::Vector{Float64} and the label of each series element;
  a difference is labelled with the row at which it ends
"""
function prepare_series(values::AbstractVector, labels::AbstractVector=1:length(values);
                        transform::Symbol=:diff, aggregate::Symbol=:none, stat::Symbol=:mean,
                        noise::Real=0.0)
    transform in TRANSFORMS || throw(ArgumentError("unknown transform $transform; use one of $TRANSFORMS"))
    aggregate in AGGREGATES || throw(ArgumentError("unknown aggregation $aggregate; use one of $AGGREGATES"))
    keep = [!ismissing(v) && v isa Real && isfinite(v) for v in values]
    v = Float64.(values[keep])
    l = collect(labels)[keep]
    if aggregate != :none
        v, l = aggregate_series(v, l; by=aggregate, stat=stat)
    end
    length(v) < 3 && throw(ArgumentError("column has fewer than 3 usable values"))

    if transform == :raw
        s = v
    elseif transform == :diff
        s, l = diff(v), l[2:end]
    else
        any(<=(0), v) && throw(ArgumentError("log returns need strictly positive values"))
        s, l = diff(log.(v)), l[2:end]
    end

    if noise > 0
        s = s .+ (noise * std(s)) .* randn(length(s))
    end
    return (series=s, labels=l)
end

end
