# Generator for the JLD2 compatibility fixtures in this directory.
#
#     julia --project test/fixtures/make_fixtures.jl
#
# `save_grid` serialises the whole `SpringsteelGridParameters` struct, so archives
# carry a record of the *value* types stored in each field. These fixtures pin the
# behaviour of `load_grid`/`_upgrade_params` (src/io.jl) against archives written by
# older versions of the struct, and exist so that compatibility is checked rather
# than assumed.
#
# ── legacy_grid_v1_rr.jld2 ────────────────────────────────────────────────────
# NOT regenerable from the current source, and deliberately so. It was written
# before `SpringsteelGridParameters` gained `num_cells_i/j/k`, so JLD2 cannot
# reconstruct the struct and hands back a `ReconstructedStatic`, exercising the
# fallback `_upgrade_params(p)` method. To reproduce it you must check out a commit
# before 58d11d8, run `save_grid` on an RR grid, and keep the blob. Do not overwrite
# it with output from this script — regenerating it under the current struct would
# silently destroy the very thing it tests.
#
# ── widened_dicts_grid.jld2 ───────────────────────────────────────────────────
# Written while the ten `Dict` fields on `SpringsteelGridParameters` were still
# declared as a *bare* `Dict` (i.e. `Dict{Any,Any}` at the type level), which allowed
# a caller to store genuinely widened dicts — `Dict{String,Any}` rather than
# `Dict{String,Int64}`. Once `vars`, `l_q`, `max_wavenumber` and the filter fields are
# narrowed to concrete parametric types, JLD2 must *convert* those on-disk values
# rather than reject them. That path is what this fixture tests; it cannot be
# regenerated once the declaration is narrowed, because the constructor now coerces
# the widened dicts on the way in.

using Springsteel
using Springsteel: SpringsteelGridParameters, createGrid, save_grid,
                   GaussianFilter, SpectralFilter, CubicBSpline

const FIXTURE_DIR = @__DIR__

function make_widened_dicts_grid()
    # Every Dict below is deliberately constructed at a *wider* type than the value
    # it holds actually needs — this is what an old archive can legally contain.
    gp = SpringsteelGridParameters(
        geometry = "RL",
        iMin = 0.0, iMax = 60.0, num_cells = 6,
        jMin = 0.0, jMax = 2π,
        vars           = Dict{String,Any}("u" => 1, "v" => 2),
        l_q            = Dict{String,Any}("default" => 2.0),
        max_wavenumber = Dict{String,Any}("default" => 4),
        BCL = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
        BCR = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
        fourier_filter = Dict{String,Any}("u" => SpectralFilter(low_pass = 3)),
        spline_filter  = Dict{String,Any}("u" => Dict{Symbol,Any}(:i => GaussianFilter(sigma = 2.0))))

    grid = createGrid(gp)

    # Put something non-trivial in the arrays so a silently-empty load is detectable.
    for v in 1:length(grid.params.vars), i in axes(grid.physical, 1)
        grid.physical[i, v, 1] = sin(0.05 * i) + 0.25 * v
    end
    grid.spectral .= 1.0

    path = joinpath(FIXTURE_DIR, "widened_dicts_grid.jld2")
    save_grid(path, grid)
    @info "wrote $path" vars = typeof(gp.vars) l_q = typeof(gp.l_q) spline_filter = typeof(gp.spline_filter)
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    if fieldtype(SpringsteelGridParameters, :vars) !== Dict
        error("""
              `vars` is declared as $(fieldtype(SpringsteelGridParameters, :vars)), not a bare Dict.
              widened_dicts_grid.jld2 can only be written while the field is still untyped —
              the narrowed constructor coerces the widened dicts on the way in, which would
              produce a fixture that tests nothing. Check out a commit before the narrowing
              to regenerate it.
              """)
    end
    make_widened_dicts_grid()
end
