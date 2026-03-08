using Springsteel
using Test
using Dates
using JLD2
using NCDatasets
using SharedArrays
using SparseArrays
using DataFrames

const TEST_GROUP = get(ENV, "TEST_GROUP", "all")

@testset "Springsteel.jl" begin
    TEST_GROUP in ("all", "basis")      && include("basis.jl")
    TEST_GROUP in ("all", "grids")      && include("grids.jl")
    TEST_GROUP in ("all", "transforms") && include("transforms.jl")
    TEST_GROUP in ("all", "tiling")     && include("tiling.jl")
    TEST_GROUP in ("all", "io")         && include("io.jl")
    TEST_GROUP in ("all", "compat")     && include("compat.jl")
    TEST_GROUP in ("all", "solver")     && include("solver.jl")
    TEST_GROUP in ("all", "mubar")      && include("mubar.jl")
end
