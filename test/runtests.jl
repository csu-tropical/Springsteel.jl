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
    TEST_GROUP in ("all", "bcholesky")  && include("banded_cholesky.jl")
    TEST_GROUP in ("all", "grids")      && include("grids.jl")
    TEST_GROUP in ("all", "transforms") && include("transforms.jl")
    TEST_GROUP in ("all", "tiling")     && include("tiling.jl")
    TEST_GROUP in ("all", "io")         && include("io.jl")
    TEST_GROUP in ("all", "solver")     && include("solver.jl")
    TEST_GROUP in ("all", "mubar")      && include("mubar.jl")
    TEST_GROUP in ("all", "interpolation") && include("interpolation.jl")
    TEST_GROUP in ("all", "filtering")      && include("filtering.jl")
    TEST_GROUP in ("all", "r3x")            && include("r3x.jl")
    TEST_GROUP in ("all", "bc")             && include("boundary_conditions.jl")
    TEST_GROUP in ("all", "multipatch")     && include("multipatch.jl")
    TEST_GROUP in ("all", "tile_multipatch") && include("tile_multipatch.jl")
end
