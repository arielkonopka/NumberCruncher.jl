using NumberCruncher
using DynamicalSystems
using Random
using Statistics
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
