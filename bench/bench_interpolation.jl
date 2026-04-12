#!/usr/bin/env julia
# Scaling benchmark for Springsteel interpolation (evaluate_unstructured,
# interpolate_to_grid) and grid relocation.
#
# Usage:
#   julia --project bench/bench_interpolation.jl [output.csv]
#
# Run before major interpolation/relocation changes to detect regressions.

using Springsteel
using BenchmarkTools
using Printf

const RESULTS = Tuple{String,String,Int,Float64,Int,Int}[]

function make_rl(nc::Int)
    gp = SpringsteelGridParameters(geometry="RL", iMin=0.0, iMax=Float64(nc),
        num_cells=nc, vars=Dict("u"=>1),
        BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
        max_wavenumber=Dict("default"=>-1))
    g = createGrid(gp)
    pts = getGridpoints(g)
    for i in 1:size(pts, 1)
        g.physical[i, 1, 1] = exp(-pts[i, 1]^2 / 16.0)
    end
    spectralTransform!(g)
    gridTransform!(g)
    return g
end

function make_rlz(nc::Int, kDim::Int)
    gp = SpringsteelGridParameters(geometry="RLZ", iMin=0.0, iMax=Float64(nc),
        kMin=0.0, kMax=10.0, num_cells=nc, kDim=kDim,
        vars=Dict("u"=>1),
        BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
        BCB=Dict("u"=>Chebyshev.R0), BCT=Dict("u"=>Chebyshev.R0),
        max_wavenumber=Dict("default"=>-1))
    g = createGrid(gp)
    pts = getGridpoints(g)
    for i in 1:size(pts, 1)
        g.physical[i, 1, 1] = exp(-pts[i, 1]^2 / 16.0) * cos(π * pts[i, 3] / 10.0)
    end
    spectralTransform!(g)
    gridTransform!(g)
    return g
end

function record(name::String, op::String, size::Int, b)
    push!(RESULTS, (name, op, size,
                    minimum(b.times) / 1e3, b.allocs, b.memory))
    @printf("  %-6s %-22s size=%-6d %10.2f μs  %d allocs  %d B\n",
        name, op, size, minimum(b.times)/1e3, b.allocs, b.memory)
end

println("── evaluate_unstructured ─────────────────────")
for nc in [10, 30]
    g = make_rl(nc)
    for npts in [100, 1000]
        pts = hcat(0.5 .+ (nc-1) .* rand(npts), 2π .* rand(npts))
        evaluate_unstructured(g, pts)  # warmup ahat cache
        b = @benchmark evaluate_unstructured($g, $pts)
        record("RL", "evaluate_unstructured", npts, b)
    end
end

for nc in [10, 20]
    g = make_rlz(nc, min(nc, 16))
    for npts in [100, 1000]
        pts = hcat(0.5 .+ (nc-1) .* rand(npts), 2π .* rand(npts), 0.5 .+ 9.0 .* rand(npts))
        evaluate_unstructured(g, pts)
        b = @benchmark evaluate_unstructured($g, $pts)
        record("RLZ", "evaluate_unstructured", npts, b)
    end
end

println("\n── relocate_grid (per-radius fast path) ──────")
for nc in [10, 30]
    g = make_rl(nc)
    relocate_grid(g, (0.5, 0.0))  # warmup
    b = @benchmark relocate_grid($g, (0.5, 0.0))
    record("RL", "relocate_grid", nc, b)
end

for nc in [10, 20]
    g = make_rlz(nc, min(nc, 16))
    relocate_grid(g, (0.5, 0.0))  # warmup
    b = @benchmark relocate_grid($g, (0.5, 0.0)) samples=5 evals=1
    record("RLZ", "relocate_grid", nc, b)
end

out = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "baseline", "interp_$(readchomp(`date +%Y-%m-%d`)).csv")
mkpath(dirname(out))
open(out, "w") do io
    println(io, "grid,op,size,time_us,allocs,mem_b")
    for (g, o, s, us, a, m) in RESULTS
        @printf(io, "%s,%s,%d,%.2f,%d,%d\n", g, o, s, us, a, m)
    end
end
println("\nCSV written to $out")
