using NumberCruncher
using DynamicalSystems
using Random
using Statistics
using Dates
using Test

const NC = NumberCruncher

# Walk-forward over the last n values of s; returns estimates, truth and last known values.
function walk_forward(s, n; k=20, msize=20, window=3000, seed=1)
    est, real, last = Float64[], Float64[], Float64[]
    for b in n:-1:1
        r = run_series(s, b; windowSize=window, msize=msize, rng=Xoshiro(hash((seed, b))))
        e = analyzeit2(r.Y, s, r.tau_vals, b; k=k, theiler=r.theiler)
        push!(est, e.center[e.searchedIndex])
        push!(real, s[end-b+1])
        push!(last, s[end-b])
    end
    return est, real, last
end

@testset "NumberCruncher.jl" begin
    lorenz = NC.DataManager.GenerateLorenz(N=5000, Δt=0.02)

    @testset "next-point coordinates" begin
        # The unknown future value sits at the largest delay, whatever the order of the delays.
        s_ext = randn(Xoshiro(3), 500)
        N = length(s_ext) - 1
        for taus in ([0, 6, 2, 8, 1, 3, 5, 4, 7], [0, 2, 4, 6, 8], [0, 9, 3], [0])
            tmax = maximum(taus)
            formula = [s_ext[N+1-(tmax-t)] for t in taus]
            @test formula ≈ collect(genembed(s_ext, taus)[end])
            @test genembed(s_ext, taus)[end][argmax(taus)] == s_ext[N+1]
        end
        # run_series's realPoint is the true next point of the reconstruction.
        s = lorenz.x
        b = 4
        r = run_series(s, b; windowSize=2000, msize=20, rng=Xoshiro(1))
        n = length(r.dataOJit)
        @test collect(genembed(s[end-b-n+1:end-b+1], r.tau_vals)[end]) ≈ r.realPoint
    end

    @testset "Theiler window and reproducibility" begin
        s = lorenz.x
        r = run_series(s, 2; windowSize=2000, msize=20, rng=Xoshiro(7))
        e = analyzeit2(r.Y, s, r.tau_vals, 2; k=25, theiler=r.theiler)
        @test r.theiler >= 1
        @test length(e.neigs.idxs) == 25
        @test all(i -> length(r.Y) - i > r.theiler, e.neigs.idxs)
        @test_throws ArgumentError analyzeit2(r.Y, s, r.tau_vals, 2; theiler=length(r.Y))
        r2 = run_series(s, 2; windowSize=2000, msize=20, rng=Xoshiro(7))
        @test r2.tau_vals == r.tau_vals && r2.dataOJit == r.dataOJit
    end

    @testset "block embeddings and PECUZAL subsampling" begin
        s = lorenz.x
        # Rebuilding the delay vectors from known delays gives exactly PECUZAL's reconstruction.
        r = run_series(s, 5; windowSize=2000, msize=20, rng=Xoshiro(2))
        r2 = run_series(s, 5; windowSize=2000, msize=20, rng=Xoshiro(2),
                        embedding=(tau_vals=r.tau_vals, theiler=r.theiler))
        @test length(r.tau_vals) > 1 && r2.Y == r.Y && r2.realPoint == r.realPoint

        # Blocks: one embedding per block, chosen on data ending at the block's largest offset.
        eb = NC.EmbeddingBlocks(s; every=10, windowSize=2000, msize=20, seed=3)
        @test NC.anchor_offset(eb, 1) == 10 && NC.anchor_offset(eb, 10) == 10 && NC.anchor_offset(eb, 11) == 20
        e1, e7 = NC.embedding_for(eb, 1), NC.embedding_for(eb, 7)
        @test e1 === e7 && length(eb.tasks) == 1                    # computed once, shared
        # no look-ahead: data from the block's first point onwards cannot change its embedding
        s2 = copy(s); s2[end-9:end] .= 0.0
        eb2 = NC.EmbeddingBlocks(s2; every=10, windowSize=2000, msize=20, seed=3)
        @test NC.embedding_for(eb2, 4).tau_vals == e1.tau_vals
        p = NC.Preprocessor.normalize_series(s)
        e = NC.estimate_point(p, 4; ks=(10,), msize=20, windowSize=2000, blocks=eb)
        @test e.anchor == 10 && e.taus == e1.tau_vals && e.real_r == s[end-3]

        # Subsampled PECUZAL is reproducible for a given seed.
        a = run_series(s, 2; windowSize=2000, msize=20, rng=Xoshiro(9), samplesize=0.3)
        b = run_series(s, 2; windowSize=2000, msize=20, rng=Xoshiro(9), samplesize=0.3)
        @test a.tau_vals == b.tau_vals && a.Y == b.Y
    end

    @testset "weights at distance zero" begin
        pts = StateSpaceSet([[1.0, 2.0], [3.0, 4.0]])
        c = NC.Analysis.weighted_center2(pts, [0.0, 1.0], Union{Float64,Nothing}[nothing, nothing])
        @test all(isfinite, c)
    end

    @testset "preparation" begin
        p = NC.Preprocessor.prepare_series([1.0, missing, 3.0, 6.0, NaN, 10.0], 'a':'f'; transform=:diff)
        @test p.series == [2.0, 3.0, 4.0]
        @test p.labels == ['c', 'd', 'f']       # a difference is labelled where it ends
        @test_throws ArgumentError NC.Preprocessor.prepare_series([1.0, -1.0, 2.0]; transform=:logdiff)
    end

    @testset "daily and weekly aggregation" begin
        # 3 full days of hourly values and a partial 4th day, which must be dropped
        t = [string(DateTime(2024, 1, 1) + Hour(h)) for h in 0:(3*24+4)]
        v = Float64.(0:(3*24+4))
        a = NC.Preprocessor.aggregate_series(v, t; by=:daily, stat=:mean)
        @test a.labels == ["2024-01-01", "2024-01-02", "2024-01-03"]
        @test a.series == [11.5, 35.5, 59.5]
        @test NC.Preprocessor.aggregate_series(v, t; by=:daily, stat=:last).series == [23.0, 47.0, 71.0]
        @test NC.Preprocessor.aggregate_series(v, t; by=:weekly).labels == ["2024-01-01"]   # a Monday
        p = NC.Preprocessor.prepare_series(v, t; aggregate=:daily, stat=:mean, transform=:diff)
        @test p.series == [24.0, 24.0] && p.labels == ["2024-01-02", "2024-01-03"]
        @test_throws ArgumentError NC.Preprocessor.prepare_series(v; aggregate=:daily)   # row numbers

        # Working effect: changes of daily means of a random walk have lag-1 autocorrelation ≈ 0.25,
        # which the AR(1) baseline picks up.
        walk = cumsum(randn(Xoshiro(5), 24 * 4000))
        hours = [string(DateTime(2000, 1, 1) + Hour(h)) for h in 0:length(walk)-1]
        d = NC.Preprocessor.prepare_series(walk, hours; aggregate=:daily, transform=:diff).series
        @test 0.18 < cor(d[1:end-1], d[2:end]) < 0.32
        @test NC.Analysis.ar1_forecast(d, 1.0) ≈ mean(d) + cor(d[1:end-1], d[2:end]) * (1.0 - mean(d))
    end

    @testset "seasonal period, differences and DST" begin
        PP = NC.Preprocessor
        hourly = [string(DateTime(2024, 1, 1) + Hour(h)) for h in 0:200]
        @test PP.detect_period(hourly) === 24
        @test PP.detect_period([string(DateTime(2024, 1, 1) + Minute(15i)) for i in 0:200]) == 96
        @test PP.detect_period([string(Date(2024, 1, 1) + Day(i)) for i in 0:50]) == 0
        @test PP.detect_period(1:100) == 0

        v = Float64.(1:60) .+ [10.0 * sin(2π * i / 24) for i in 1:60]    # trend + daily cycle
        p = PP.prepare_series(v, 1:60; transform=:sdiff, period=24)
        @test p.series ≈ fill(24.0, 36)                                  # the cycle is gone
        @test p.labels == 25:60
        @test_throws ArgumentError PP.prepare_series(v; transform=:sdiff)    # needs a period

        # UTC labels: every lag of 24 samples is exactly one day
        @test PP.seasonal_regularity(hourly, 24).exact == 1.0
        # Local time with DST (CET/CEST): spring skips 02:00, autumn repeats it. The sampling is
        # still hourly, so the lags span 24 real hours; they show up as DST, not as gaps.
        utc = [DateTime(2024, 3, 25) + Hour(h) for h in 0:(24*230)]
        summer(t) = DateTime(2024, 3, 31, 1) <= t < DateTime(2024, 10, 27, 1)
        loc = [string(t + Hour(summer(t) ? 2 : 1)) for t in utc]
        reg = PP.seasonal_regularity(loc, 24)
        @test reg.other == 0.0 && reg.dst > 0 && reg.exact + reg.dst ≈ 1.0
        @test PP.detect_period(loc) == 24
        # a real gap (a missing hour) is reported as such
        gappy = vcat(hourly[1:100], hourly[102:end])
        @test PP.seasonal_regularity(gappy, 24).other > 0

        # the seasonal persistence baseline is the value one period earlier
        s = randn(Xoshiro(8), 3000)
        sp = PP.normalize_series(s, [string(DateTime(2024, 1, 1) + Hour(h)) for h in 0:2999]; period=24)
        e = NC.estimate_point(sp, 5; ks=(10,), msize=5, windowSize=1500)
        @test e.seasonal_r == s[end-4-24]
    end

    @testset "volatility normalisation" begin
        PP = NC.Preprocessor
        r = randn(Xoshiro(21), 3000)
        # No look-ahead: changing r[t] leaves σ[1..t] unchanged and only moves σ[t+1] onwards.
        for (name, vol) in (("ewma", x -> PP.ewma_volatility(x; λ=0.98, init_n=500)),
                            ("rolling", x -> PP.rolling_volatility(x; W=168)))
            t = 1200
            r2 = copy(r); r2[t] += 10.0
            σ, σ2 = vol(r), vol(r2)
            @test isequal(σ[1:t], σ2[1:t])
            @test σ2[t+1] != σ[t+1]
        end

        # r = s·ε with a slowly varying scale s: z = r/σ has std ≈ 1 in calm and turbulent stretches.
        n = 40_000
        s = [1.0 + 9.0 * (sin(2π * i / 10_000) + 1) / 2 for i in 1:n]   # scale 1 … 10
        x = s .* randn(Xoshiro(22), n)
        for normalize in (:ewma, :rolling)
            p = PP.normalize_series(x; normalize=normalize)
            calm = findall(<(2.0), s[p.vol_burnin+1:end])
            wild = findall(>(9.0), s[p.vol_burnin+1:end])
            @test 0.9 < std(p.z[calm]) < 1.1
            @test 0.9 < std(p.z[wild]) < 1.1
            @test std(x[p.vol_burnin.+wild]) / std(x[p.vol_burnin.+calm]) > 5    # the raw scales really differ
            @test p.r .* 1 ≈ p.z .* p.scale                                       # back-conversion
        end

        # normalize = :none changes nothing: same values, unit scale, no burn-in.
        p = PP.normalize_series(r; normalize=:none)
        @test p.z == r && all(==(1.0), p.scale) && p.vol_burnin == 0
        @test_throws ArgumentError PP.normalize_series(r; normalize=:garch)
    end

    @testset "estimate_point variants" begin
        s = lorenz.x
        p = NC.Preprocessor.normalize_series(s; normalize=:none)
        e = NC.estimate_point(p, 3; ks=(15,), msize=20, windowSize=2000, seed=4)
        # bit-identical to calling the reconstruction and analyzeit2 directly with the same seed
        r = run_series(s, 3; windowSize=2000, msize=20, rng=Xoshiro(hash((4, 3))))
        a = analyzeit2(r.Y, s, r.tau_vals, 3; k=15, theiler=r.theiler)
        @test e.est["15"].ret === a.center[a.searchedIndex]
        @test e.real_r == s[end-2] && e.scale == 1.0
        # shadow off = plain inverse-distance weights of the successors
        w = 1 ./ (a.neigs.dists .+ 1e-12 * max(1.0, maximum(a.neigs.dists)))
        @test e.est["15"].ret_ns ≈ sum(w .* [q[a.searchedIndex] for q in a.successors]) / sum(w)
        @test e.est["15"].abs ≥ 0 && e.est["15"].abs_ns ≥ 0
    end

    @testset "binomial p-value" begin
        @test NC.Analysis.binomial_pvalue(5, 10) == 1.0
        @test NC.Analysis.binomial_pvalue(0, 10) ≈ 2 / 1024
        @test NC.Analysis.binomial_pvalue(10, 10) ≈ 2 / 1024
    end

    # Sanity checks of the method: deterministic chaos is predictable, white noise is not.
    @testset "skill: Lorenz vs white noise" begin
        est, real, last = walk_forward(lorenz.x, 15)
        sk = NC.Analysis.skill_summary(est, real, last)
        @test sk.skill_persistence > 0.5        # far better than repeating the last value
        @test sk.skill_zero > 0.9

        noise = randn(Xoshiro(11), 4000)
        est, real, last = walk_forward(noise, 15; msize=10)
        sk = NC.Analysis.skill_summary(est, real, last)
        @test sk.skill_zero < 0.15              # no real skill over predicting the mean
    end
end
