#Functions for I/O

# ════════════════════════════════════════════════════════════════════════════
# Generic SpringsteelGrid I/O (Phase 9)
# ════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# check_grid_dims — generic SpringsteelGrid
# ────────────────────────────────────────────────────────────────────────────

"""
    check_grid_dims(physical_data::DataFrame, grid::SpringsteelGrid)

Verify that the number of rows in `physical_data` matches the physical-array
row count of `grid` (`size(grid.physical, 1)`).

Throws a `DomainError` when sizes do not match, with the actual row count as
the offending value.  Returns `nothing` on success.

# Arguments
- `physical_data`: A `DataFrame` whose rows correspond to physical gridpoints.
- `grid`: Any [`SpringsteelGrid`](@ref) instance.

# Throws
- `DomainError(actual, msg)` if `nrow(physical_data) ≠ size(grid.physical, 1)`.

See also: [`read_physical_grid`](@ref), [`write_grid`](@ref)
"""
function check_grid_dims(physical_data::DataFrame, grid::SpringsteelGrid)
    expected = size(grid.physical, 1)
    actual   = nrow(physical_data)
    if expected != actual
        throw(DomainError(actual,
            "Grid size does not match: expected $expected rows, got $actual"))
    end
end

# ────────────────────────────────────────────────────────────────────────────
# _io_vars_sorted — shared helper: variables sorted by index
# ────────────────────────────────────────────────────────────────────────────
_io_vars_sorted(gp) = sort(collect(pairs(gp.vars)), by = x -> x[2])

# ────────────────────────────────────────────────────────────────────────────
# write_grid — 1D (NoBasisArray j and k)
# ────────────────────────────────────────────────────────────────────────────

"""
    write_grid(grid::SpringsteelGrid{G, I, NoBasisArray, NoBasisArray}, output_dir, tag)

Write spectral coefficients, physical-space values, and a regular-grid
interpolant for a 1-D [`SpringsteelGrid`](@ref) to three CSV files in
`output_dir`:

| File | Contents |
|:---- |:-------- |
| `\$(tag)_spectral.csv`  | columns `r`, then variable names; `b_iDim` rows |
| `\$(tag)_physical.csv`  | columns `r`, then `<var>`, `<var>_r`, `<var>_rr`; `iDim` rows |
| `\$(tag)_gridded.csv`   | same column layout as physical but on regular grid |

# Arguments
- `grid`: 1-D [`SpringsteelGrid`](@ref) with an up-to-date `physical` (for the
  regular-grid output, the spectral array must also be current).
- `output_dir::String`: Directory path for output files (must exist).
- `tag::String`: Filename prefix string.

See also: [`getGridpoints`](@ref), [`getRegularGridpoints`](@ref),
[`regularGridTransform`](@ref)
"""
function write_grid(grid::SpringsteelGrid{G, I, NoBasisArray, NoBasisArray},
                    output_dir::String, tag::String) where {G, I}
    println("Writing $tag to $output_dir")
    suffix     = ["", "_r", "_rr"]
    vars_s     = _io_vars_sorted(grid.params)
    nderiv     = 3

    # ── spectral file ─────────────────────────────────────────────────────
    open(joinpath(output_dir, "$(tag)_spectral.csv"), "w") do af
        hdr = "r"
        for (v, _) in vars_s; hdr *= ",$v"; end
        println(af, hdr)
        for r in 1:grid.params.b_iDim
            row = "$r"
            for (_, vi) in vars_s; row *= ",$(grid.spectral[r, vi])"; end
            println(af, row)
        end
    end

    # ── physical file ─────────────────────────────────────────────────────
    gridpts = getGridpoints(grid)
    open(joinpath(output_dir, "$(tag)_physical.csv"), "w") do uf
        hdr = "r"
        for d in 1:nderiv
            for (v, _) in vars_s; hdr *= ",$(v)$(suffix[d])"; end
        end
        println(uf, hdr)
        for r in 1:grid.params.iDim
            row = "$(gridpts[r])"
            for d in 1:nderiv
                for (_, vi) in vars_s; row *= ",$(grid.physical[r, vi, d])"; end
            end
            println(uf, row)
        end
    end

    # ── regular grid file ─────────────────────────────────────────────────
    reg_pts  = getRegularGridpoints(grid)
    reg_phys = regularGridTransform(grid, reg_pts)
    open(joinpath(output_dir, "$(tag)_gridded.csv"), "w") do rf
        hdr = "r"
        for d in 1:nderiv
            for (v, _) in vars_s; hdr *= ",$(v)$(suffix[d])"; end
        end
        println(rf, hdr)
        for r in eachindex(reg_pts)
            row = "$(reg_pts[r])"
            for d in 1:nderiv
                for (_, vi) in vars_s; row *= ",$(reg_phys[r, vi, d])"; end
            end
            println(rf, row)
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────
# write_grid — 2D j-active (RL, RR, SL; k slot is NoBasisArray)
# ────────────────────────────────────────────────────────────────────────────

"""
    write_grid(grid::SpringsteelGrid{G, I, J, NoBasisArray}, output_dir, tag)

Write spectral coefficients and physical-space values for a 2-D
[`SpringsteelGrid`](@ref) with an active j-dimension (e.g. `RL_Grid`,
`RR_Grid`, `SL_Grid`) to two CSV files:

| File | Contents |
|:---- |:-------- |
| `\$(tag)_spectral.csv`  | flat spectral indices with variable values |
| `\$(tag)_physical.csv`  | columns `r, l`, then 5 derivative slots per variable |

See also: [`write_grid`](@ref)
"""
function write_grid(grid::SpringsteelGrid{G, I, J, NoBasisArray},
                    output_dir::String, tag::String) where {G, I,
                    J <: Union{SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}}
    println("Writing $tag to $output_dir")
    suffix = ["", "_r", "_rr", "_l", "_ll"]
    vars_s = _io_vars_sorted(grid.params)
    nderiv = 5

    # ── spectral file ─────────────────────────────────────────────────────
    open(joinpath(output_dir, "$(tag)_spectral.csv"), "w") do af
        hdr = "idx"
        for (v, _) in vars_s; hdr *= ",$v"; end
        println(af, hdr)
        for r in 1:size(grid.spectral, 1)
            row = "$r"
            for (_, vi) in vars_s; row *= ",$(grid.spectral[r, vi])"; end
            println(af, row)
        end
    end

    # ── physical file ─────────────────────────────────────────────────────
    gridpts = getGridpoints(grid)   # Matrix{Float64}: (npoints, 2)
    open(joinpath(output_dir, "$(tag)_physical.csv"), "w") do uf
        hdr = "r,l"
        for d in 1:nderiv
            for (v, _) in vars_s; hdr *= ",$(v)$(suffix[d])"; end
        end
        println(uf, hdr)
        n = size(gridpts, 1)
        for p in 1:n
            row = "$(gridpts[p, 1]),$(gridpts[p, 2])"
            for d in 1:nderiv
                for (_, vi) in vars_s; row *= ",$(grid.physical[p, vi, d])"; end
            end
            println(uf, row)
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────
# write_grid — 2D k-active (RZ; j slot is NoBasisArray)
# ────────────────────────────────────────────────────────────────────────────

"""
    write_grid(grid::SpringsteelGrid{G, I, NoBasisArray, K}, output_dir, tag)

Write spectral coefficients and physical-space values for a 2-D
[`SpringsteelGrid`](@ref) with an active k-dimension (e.g. `RZ_Grid`) to two
CSV files:

| File | Contents |
|:---- |:-------- |
| `\$(tag)_spectral.csv`  | flat spectral indices with variable values |
| `\$(tag)_physical.csv`  | columns `r, z`, then 5 derivative slots per variable |

See also: [`write_grid`](@ref)
"""
function write_grid(grid::SpringsteelGrid{G, I, NoBasisArray, K},
                    output_dir::String, tag::String) where {G, I,
                    K <: Union{SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}}
    println("Writing $tag to $output_dir")
    suffix = ["", "_r", "_rr", "_z", "_zz"]
    vars_s = _io_vars_sorted(grid.params)
    nderiv = 5

    # ── spectral file ─────────────────────────────────────────────────────
    open(joinpath(output_dir, "$(tag)_spectral.csv"), "w") do af
        hdr = "idx"
        for (v, _) in vars_s; hdr *= ",$v"; end
        println(af, hdr)
        for r in 1:size(grid.spectral, 1)
            row = "$r"
            for (_, vi) in vars_s; row *= ",$(grid.spectral[r, vi])"; end
            println(af, row)
        end
    end

    # ── physical file ─────────────────────────────────────────────────────
    gridpts = getGridpoints(grid)   # Matrix{Float64}: (npoints, 2)
    open(joinpath(output_dir, "$(tag)_physical.csv"), "w") do uf
        hdr = "r,z"
        for d in 1:nderiv
            for (v, _) in vars_s; hdr *= ",$(v)$(suffix[d])"; end
        end
        println(uf, hdr)
        n = size(gridpts, 1)
        for p in 1:n
            row = "$(gridpts[p, 1]),$(gridpts[p, 2])"
            for d in 1:nderiv
                for (_, vi) in vars_s; row *= ",$(grid.physical[p, vi, d])"; end
            end
            println(uf, row)
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────
# write_grid — 3D (both j and k active: RLZ, RRR, SLZ)
# ────────────────────────────────────────────────────────────────────────────

"""
    write_grid(grid::SpringsteelGrid{G, I, J, K}, output_dir, tag)

Write spectral coefficients and physical-space values for a 3-D
[`SpringsteelGrid`](@ref) (e.g. `RLZ_Grid`, `RRR_Grid`, `SLZ_Grid`) to two
CSV files:

| File | Contents |
|:---- |:-------- |
| `\$(tag)_spectral.csv`  | flat spectral indices with variable values |
| `\$(tag)_physical.csv`  | columns `r, l, z`, then 7 derivative slots per variable |

See also: [`write_grid`](@ref)
"""
function write_grid(grid::SpringsteelGrid{G, I, J, K},
                    output_dir::String, tag::String) where {G, I,
                    J <: Union{SplineBasisArray, FourierBasisArray, ChebyshevBasisArray},
                    K <: Union{SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}}
    println("Writing $tag to $output_dir")
    suffix = ["", "_r", "_rr", "_l", "_ll", "_z", "_zz"]
    vars_s = _io_vars_sorted(grid.params)
    nderiv = 7

    # ── spectral file ─────────────────────────────────────────────────────
    open(joinpath(output_dir, "$(tag)_spectral.csv"), "w") do af
        hdr = "idx"
        for (v, _) in vars_s; hdr *= ",$v"; end
        println(af, hdr)
        for r in 1:size(grid.spectral, 1)
            row = "$r"
            for (_, vi) in vars_s; row *= ",$(grid.spectral[r, vi])"; end
            println(af, row)
        end
    end

    # ── physical file ─────────────────────────────────────────────────────
    gridpts = getGridpoints(grid)   # Matrix{Float64}: (npoints, 3)
    open(joinpath(output_dir, "$(tag)_physical.csv"), "w") do uf
        hdr = "r,l,z"
        for d in 1:nderiv
            for (v, _) in vars_s; hdr *= ",$(v)$(suffix[d])"; end
        end
        println(uf, hdr)
        n = size(gridpts, 1)
        for p in 1:n
            row = "$(gridpts[p, 1]),$(gridpts[p, 2]),$(gridpts[p, 3])"
            for d in 1:nderiv
                for (_, vi) in vars_s; row *= ",$(grid.physical[p, vi, d])"; end
            end
            println(uf, row)
        end
    end
end

# ════════════════════════════════════════════════════════════════════════════
# Legacy per-type methods (old struct types — R_Grid, RL_Grid, RZ_Grid, RLZ_Grid)
# ════════════════════════════════════════════════════════════════════════════

"""
    read_physical_grid(file::String, grid::AbstractGrid)

Read physical-space values from `file` (CSV format) into `grid`.

The CSV must have one column per variable (named with the variable key) and
one row per physical gridpoint.  The function calls [`check_grid_dims`](@ref)
to verify row counts before populating `grid.physical[:, v, 1]` for each
variable `v`.

# Arguments
- `file`: Path to the CSV file.
- `grid`: Any grid implementing the `AbstractGrid` interface.

# Throws
- `DomainError` if the CSV row count does not match the grid's physical array.
- `DomainError` if a required variable column is absent from the CSV.

See also: [`write_grid`](@ref), [`check_grid_dims`](@ref)
"""
function read_physical_grid(file::String, grid::AbstractGrid)

    # Initialize the patch on each process
    physical_data = CSV.read(file, DataFrame, header=1)
    
    # Check that the dimensions are correct
    check_grid_dims(physical_data, grid)
    
    # Check for all the variables
    for key in keys(grid.params.vars)
        foundkey = false
        for name in names(physical_data)
            if (name == key)
                foundkey = true
            end
        end
        if foundkey == false
            throw(DomainError(key, "Grid is missing variable"))
        end
    end
    
    # Assign variables
    for (key, value) in pairs(grid.params.vars)
        grid.physical[:,value,1] .= select(physical_data, key)
    end
end


