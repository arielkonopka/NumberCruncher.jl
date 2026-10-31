module GUI
# Browser GUI: a small JSON API over HTTP.jl plus the static page in ./static.
# Heavy work (PECUZAL, downloads) runs on the default threadpool; the server
# stays on the interactive one so the page keeps responding during a run.
using HTTP
using JSON
using CSV
using DataFrames
using Dates
using Statistics
using Random
using Base.Threads

const NC = parentmodule(@__MODULE__)
const DM = NC.DataManager
const PP = NC.Preprocessor

const DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "data"))
const STATIC_DIR = joinpath(@__DIR__, "static")
const FILE_RE = r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.csv$"

# ==============================================================================
# STATE
# ==============================================================================
mutable struct Dataset
    file::String
    column::String
    transform::Symbol
    noise::Float64
    series::Vector{Float64}
    labels::Vector{String}
end

mutable struct RunJob
    params::Dict{String,Any}
    dataset::Dataset
    total::Int
    done::Atomic{Int}
    failed::Atomic{Int}
    cancel::Atomic{Bool}
    status::Symbol              # :running, :done, :cancelled, :failed
    message::String
    results::Vector{NamedTuple}
    started::DateTime
    finished::Union{DateTime,Nothing}
    lock::ReentrantLock
end

const STATE_LOCK = ReentrantLock()
const DATASET = Ref{Union{Dataset,Nothing}}(nothing)
const JOB = Ref{Union{RunJob,Nothing}}(nothing)
const FETCH = Ref{Any}((status="idle", message="", file=""))
const FILE_INFO = Dict{String,Tuple{Float64,Any}}()     # path => (mtime, info)

# ==============================================================================
# HELPERS
# ==============================================================================
json_response(x; status=200) =
    HTTP.Response(status, ["Content-Type" => "application/json"], JSON.json(x))

body(req) = isempty(req.body) ? Dict{String,Any}() : JSON.parse(String(req.body))

getnum(p, key, default, T=Float64) = haskey(p, key) && p[key] !== nothing ? T(p[key]) : T(default)

# Runs f on a worker thread and waits without blocking the server thread.
function onworker(f)
    try
        return fetch(Threads.@spawn f())
    catch e
        throw(e isa TaskFailedException ? e.task.exception : e)
    end
end

function data_path(name)
    occursin(FILE_RE, name) || throw(ArgumentError("invalid file name '$name' (letters, digits, _ . -, ending in .csv)"))
    return joinpath(DATA_DIR, name)
end

function file_info(path)
    mt = mtime(path)
    cached = get(FILE_INFO, path, nothing)
    cached !== nothing && cached[1] == mt && return cached[2]
    df = DM.LoadData(path)
    labels = DM.RowLabels(df)
    info = (name=basename(path), rows=nrow(df), columns=DM.NumericColumns(df),
            first=isempty(labels) ? "" : labels[1], last=isempty(labels) ? "" : labels[end],
            modified=string(unix2datetime(mt)))
    FILE_INFO[path] = (mt, info)
    return info
end

function list_files()
    isdir(DATA_DIR) || return []
    files = sort(filter(f -> occursin(FILE_RE, f), readdir(DATA_DIR)))
    out = []
    for f in files
        try
            push!(out, file_info(joinpath(DATA_DIR, f)))
        catch e
            push!(out, (name=f, error=sprint(showerror, e)))
        end
    end
    return out
end

downsample(n, maxn) = 1:max(1, cld(n, maxn)):n

function dataset_summary(d::Dataset)
    s = d.series
    idx = downsample(length(s), 3000)
    return (file=d.file, column=d.column, transform=string(d.transform), noise=d.noise,
            length=length(s), mean=mean(s), std=std(s), min=minimum(s), max=maximum(s),
            first=d.labels[1], last=d.labels[end],
            preview=(index=collect(idx), labels=d.labels[idx], values=s[idx]))
end

# ==============================================================================
# ESTIMATION
# ==============================================================================
# The jitter is seeded per offset, so a run is reproducible and independent of thread order.
function evaluate_point(d::Dataset, b, ks, msize, window, seed)
    s = d.series
    r0 = NC.run_series(s, b; windowSize=window, msize=msize, rng=Xoshiro(hash((seed, b))))
    est = Dict{String,Float64}()
    for k in ks
        r1 = NC.analyzeit2(r0.Y, s, r0.tau_vals, b; k=k, theiler=r0.theiler)
        est[string(k)] = r1.center[r1.searchedIndex]
    end
    return (offset=b, label=d.labels[end-b+1], m=length(r0.tau_vals), taus=r0.tau_vals, theiler=r0.theiler,
            real=s[end-b+1], last=s[end-b], window_mean=mean(r0.dataOJit), est=est)
end

function run_job!(job::RunJob, ks, msize, window, seed)
    n = job.total
    try
        Threads.@threads for b in n:-1:1
            job.cancel[] && continue
            try
                r = evaluate_point(job.dataset, b, ks, msize, window, seed)
                lock(job.lock) do
                    push!(job.results, r)
                end
            catch e
                atomic_add!(job.failed, 1)
                lock(job.lock) do
                    job.message = "offset $b: " * sprint(showerror, e)
                end
            end
            atomic_add!(job.done, 1)
        end
        job.status = job.cancel[] ? :cancelled : :done
    catch e
        job.status = :failed
        job.message = sprint(showerror, e)
    end
    job.finished = now()
end

nanless(x) = x === nothing || (x isa Real && !isfinite(x)) ? nothing : x

function score(e, real, last, raw)
    err = e .- real
    dir(x) = raw ? sign.(x .- last) : sign.(x)
    de, dr = dir(e), dir(real)
    ok = dr .!= 0
    defined = !all(==(0), de) && any(ok)
    hits = count(de[ok] .== dr[ok])
    hit = defined ? 100 * hits / count(ok) : nothing
    hit_p = defined ? NC.Analysis.binomial_pvalue(hits, count(ok)) : nothing
    c = length(e) > 2 && std(e) > 0 && std(real) > 0 ? cor(e, real) : nothing
    return (mae=mean(abs.(err)), rmse=sqrt(mean(err .^ 2)), hit=nanless(hit), hit_p=hit_p, corr=nanless(c))
end

# Everything is compared with the best naive baseline: skill > 0 beats all of them.
function metrics(res, ks, raw)
    isempty(res) && return (models=[], by_m=[], best_baseline=nothing)
    real = [r.real for r in res]
    last = [r.last for r in res]
    baselines = ["persistence" => last, "window mean" => [r.window_mean for r in res]]
    # Predicting 0 only makes sense for changes, not for raw levels like a temperature.
    raw || pushfirst!(baselines, "zero" => zeros(length(res)))
    knn = ["k = $k" => [r.est[string(k)] for r in res] for k in ks]
    scored = [(model=name, baseline=i > length(knn), score(e, real, last, raw)...)
              for (i, (name, e)) in enumerate(vcat(knn, baselines))]
    best = argmin(r -> r.mae, filter(r -> r.baseline, scored))
    models = [merge(r, (skill=1 - r.mae / best.mae,)) for r in scored]

    by_m = []
    for m in sort(unique(r.m for r in res))
        g = filter(r -> r.m == m, res)
        gr, gl = [r.real for r in g], [r.last for r in g]
        push!(by_m, (m=m, n=length(g),
            knn=Dict(string(k) => score([r.est[string(k)] for r in g], gr, gl, raw) for k in ks),
            persistence_mae=mean(abs.(gl .- gr)), zero_mae=mean(abs.(gr))))
    end
    return (models=models, by_m=by_m, best_baseline=best.model)
end

function job_status(job)
    job === nothing && return (status="idle",)
    res = lock(job.lock) do
        sort(copy(job.results), by=r -> r.offset, rev=true)
    end
    ks = job.params["ks"]
    raw = job.dataset.transform == :raw
    finished = something(job.finished, now())
    return (status=string(job.status), message=job.message, params=job.params,
            dataset=(file=job.dataset.file, column=job.dataset.column,
                     transform=string(job.dataset.transform), noise=job.dataset.noise),
            total=job.total, done=job.done[], failed=job.failed[],
            elapsed=Dates.value(finished - job.started) / 1000,
            raw=raw, metrics=metrics(res, ks, raw), results=res)
end

function phase_space(d::Dataset, b, msize, window, k, seed)
    s = d.series
    r0 = NC.run_series(s, b; windowSize=window, msize=msize, rng=Xoshiro(hash((seed, b))))
    Y = r0.Y
    r1 = NC.analyzeit2(Y, s, r0.tau_vals, b; k=k, theiler=r0.theiler)
    idx = downsample(length(Y), 4000)
    cols(pts) = [[p[j] for p in pts] for j in 1:length(r0.tau_vals)]
    return (taus=r0.tau_vals, theiler=r0.theiler, m=length(r0.tau_vals), n=length(Y),
            searched_index=r1.searchedIndex, label=d.labels[end-b+1],
            trajectory=cols(Y[idx]), neighbours=cols(Y[r1.neigs.idxs]),
            successors=cols(Y[r1.neigs.idxs .+ 1]),
            last=collect(Y[end]), estimate=r1.center, real=r0.realPoint)
end

# ==============================================================================
# HANDLERS
# ==============================================================================
function get_data(req)
    files = onworker(list_files)
    d = DATASET[]
    return json_response((data_dir=DATA_DIR, files=files, transforms=string.(PP.TRANSFORMS),
                          meteo_variables=DM.METEO_VARIABLES, yahoo_intervals=DM.YAHOO_INTERVALS,
                          dataset=d === nothing ? nothing : dataset_summary(d),
                          fetch=FETCH[], key_present=isfile(NC.CRYPTOCOMPARE_KEY_FILE)))
end

function post_dataset(req)
    p = body(req)
    path = data_path(p["file"])
    column = string(p["column"])
    transform = Symbol(get(p, "transform", "diff"))
    noise = getnum(p, "noise", 0.0)
    0 <= noise <= 10 || throw(ArgumentError("noise must be between 0 and 10 (fraction of std)"))
    d = onworker() do
        df = DM.LoadData(path)
        column in names(df) || throw(ArgumentError("column '$column' not in $(basename(path))"))
        prep = PP.prepare_series(df[!, column], DM.RowLabels(df); transform=transform, noise=noise)
        Dataset(basename(path), column, transform, noise, prep.series, prep.labels)
    end
    DATASET[] = d
    return json_response(dataset_summary(d))
end

function post_lorenz(req)
    p = body(req)
    path = data_path(p["name"])
    conflict = exists_conflict(path, p)
    conflict === nothing || return conflict
    N = getnum(p, "N", 20000, Int)
    1000 <= N <= 2_000_000 || throw(ArgumentError("N must be between 1000 and 2000000"))
    df = onworker() do
        DM.GenerateLorenz(N=N, Δt=getnum(p, "dt", 0.01), Ttr=getnum(p, "Ttr", 100.0),
                          σ=getnum(p, "sigma", 10.0), ρ=getnum(p, "rho", 28.0), β=getnum(p, "beta", 8/3),
                          u0=[getnum(p, "x0", 0.0), getnum(p, "y0", 10.0), getnum(p, "z0", 0.0)])
    end
    mkpath(DATA_DIR)
    CSV.write(path, df)
    return json_response((ok=true, file=basename(path), rows=nrow(df)))
end

function exists_conflict(path, p)
    isfile(path) && get(p, "overwrite", false) != true || return nothing
    return json_response((error="$(basename(path)) exists", exists=true); status=409)
end

function post_import(req)
    p = body(req)
    path = data_path(p["name"])
    conflict = exists_conflict(path, p)
    conflict === nothing || return conflict
    source = p["source"]
    df = onworker() do
        if source == "yahoo"
            DM.FetchYahoo(strip(p["symbol"]); interval=get(p, "interval", "1d"))
        elseif source == "fred"
            DM.FetchFRED(strip(p["series"]))
        elseif source == "openmeteo"
            DM.FetchOpenMeteo(getnum(p, "latitude", 0.0), getnum(p, "longitude", 0.0),
                              Date(p["start_date"]), Date(p["end_date"]);
                              variables=String.(get(p, "variables", ["temperature_2m"])))
        else
            throw(ArgumentError("unknown source '$source'"))
        end
    end
    mkpath(DATA_DIR)
    CSV.write(path, df)
    return json_response((ok=true, file=basename(path), rows=nrow(df)))
end

const MAX_UPLOAD = 200 * 1024^2

function post_upload(req)
    q = HTTP.queryparams(HTTP.URI(req.target))
    path = data_path(get(q, "name", ""))
    conflict = exists_conflict(path, Dict("overwrite" => get(q, "overwrite", "") == "true"))
    conflict === nothing || return conflict
    length(req.body) <= MAX_UPLOAD || throw(ArgumentError("file larger than 200 MB"))
    df = onworker(() -> CSV.read(req.body, DataFrame))
    isempty(DM.NumericColumns(df)) && throw(ArgumentError("the CSV has no numeric column"))
    mkpath(DATA_DIR)
    write(path, req.body)
    return json_response((ok=true, file=basename(path), rows=nrow(df), columns=DM.NumericColumns(df)))
end

function post_fetch(req)
    p = body(req)
    path = data_path(get(p, "name", "btc_cryptocompare.csv"))
    FETCH[].status == "running" && return json_response((error="a download is already running",); status=409)
    conflict = exists_conflict(path, p)
    conflict === nothing || return conflict
    start = Date(get(p, "start_date", "2008-01-01"))
    FETCH[] = (status="running", message="downloading…", file=basename(path))
    Threads.@spawn begin
        r = try
            NC.update_from_cryptocompare!(path; start_date=DateTime(start))
        catch e
            (ok=false, rows=0, message=sprint(showerror, e))
        end
        FETCH[] = (status=r.ok ? "done" : "failed",
                   message=r.ok ? "$(r.rows) rows written ($(r.message))" : r.message,
                   file=basename(path))
    end
    return json_response(FETCH[])
end

function post_run(req)
    p = body(req)
    d = DATASET[]
    d === nothing && throw(ArgumentError("choose a dataset first"))
    lock(STATE_LOCK) do
        j = JOB[]
        j !== nothing && j.status == :running && throw(ArgumentError("a run is already in progress"))
        n = getnum(p, "n_points", 100, Int)
        ks = sort(unique(Int.(get(p, "ks", [50]))))
        msize = getnum(p, "msize", 10, Int)
        window = getnum(p, "window", 82000, Int)
        seed = getnum(p, "seed", 1, Int)
        1 <= n <= length(d.series) ÷ 2 || throw(ArgumentError("points to test must be between 1 and $(length(d.series) ÷ 2)"))
        !isempty(ks) && all(>=(1), ks) || throw(ArgumentError("k values must be positive"))
        1 <= msize <= 200 || throw(ArgumentError("max delay must be between 1 and 200"))
        window >= 100 || throw(ArgumentError("window must be at least 100"))
        params = Dict{String,Any}("n_points" => n, "ks" => ks, "msize" => msize, "window" => window, "seed" => seed)
        job = RunJob(params, d, n, Atomic{Int}(0), Atomic{Int}(0), Atomic{Bool}(false),
                     :running, "", NamedTuple[], now(), nothing, ReentrantLock())
        JOB[] = job
        Threads.@spawn run_job!(job, ks, msize, window, seed)
    end
    return json_response(job_status(JOB[]))
end

function post_stop(req)
    j = JOB[]
    j !== nothing && (j.cancel[] = true)
    return json_response((ok=true,))
end

get_status(req) = json_response(job_status(JOB[]))

function post_phase(req)
    p = body(req)
    d = DATASET[]
    d === nothing && throw(ArgumentError("choose a dataset first"))
    b = getnum(p, "offset", 1, Int)
    1 <= b <= length(d.series) ÷ 2 || throw(ArgumentError("offset out of range"))
    r = onworker(() -> phase_space(d, b, getnum(p, "msize", 10, Int), getnum(p, "window", 82000, Int),
                                   getnum(p, "k", 50, Int), getnum(p, "seed", 1, Int)))
    return json_response(r)
end

function static_file(name, mime)
    return req -> HTTP.Response(200, ["Content-Type" => mime], read(joinpath(STATIC_DIR, name)))
end

# PlotlyJS.jl ships plotly.js, so the page works offline.
function plotly_js(req)
    path = NC.PlotlyJS._js_path
    isfile(path) || return HTTP.Response(404, "plotly.js not found")
    return HTTP.Response(200, ["Content-Type" => "application/javascript"], read(path))
end

function guarded(f)
    return function (req)
        try
            return f(req)
        catch e
            e = e isa TaskFailedException ? e.task.exception : e
            status = e isa ArgumentError || e isa KeyError ? 400 : 500
            return json_response((error=sprint(showerror, e),); status=status)
        end
    end
end

function router()
    r = HTTP.Router()
    HTTP.register!(r, "GET", "/", static_file("index.html", "text/html; charset=utf-8"))
    HTTP.register!(r, "GET", "/app.js", static_file("app.js", "application/javascript"))
    HTTP.register!(r, "GET", "/plotly.min.js", plotly_js)
    HTTP.register!(r, "GET", "/api/data", guarded(get_data))
    HTTP.register!(r, "POST", "/api/dataset", guarded(post_dataset))
    HTTP.register!(r, "POST", "/api/data/lorenz", guarded(post_lorenz))
    HTTP.register!(r, "POST", "/api/data/fetch", guarded(post_fetch))
    HTTP.register!(r, "POST", "/api/data/import", guarded(post_import))
    HTTP.register!(r, "POST", "/api/data/upload", guarded(post_upload))
    HTTP.register!(r, "POST", "/api/run", guarded(post_run))
    HTTP.register!(r, "POST", "/api/run/stop", guarded(post_stop))
    HTTP.register!(r, "GET", "/api/run/status", guarded(get_status))
    HTTP.register!(r, "POST", "/api/phase", guarded(post_phase))
    return r
end

"""
    serve_gui(; host="127.0.0.1", port=8080, block=true)

Starts the browser GUI. Start Julia with several threads (e.g. `julia -t 6`);
the GUI only listens on localhost by default because it has no authentication.
"""
function serve_gui(; host="127.0.0.1", port=8080, block=true)
    Threads.nthreads(:default) < 2 &&
        @warn "Only one worker thread; start Julia with e.g. `julia -t 6` for parallel runs."
    r = router()
    server = Threads.nthreads(:interactive) > 0 ?
        fetch(Threads.@spawn :interactive HTTP.serve!(r, host, port)) :
        HTTP.serve!(r, host, port)
    println("NumberCruncher GUI running at http://$host:$port  (Ctrl+C to stop)")
    block || return server
    try
        wait(server)
    catch e
        e isa InterruptException || rethrow()
    finally
        close(server)
    end
end

end
