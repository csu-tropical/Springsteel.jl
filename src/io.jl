#Functions for I/O

# ════════════════════════════════════════════════════════════════════════════
# Generic SpringsteelGrid I/O
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

    # ── regular grid file ─────────────────────────────────────────────────
    reg_pts  = getRegularGridpoints(grid)   # Matrix{Float64}: (n_r*n_λ, 2)
    reg_phys = regularGridTransform(grid, reg_pts)
    open(joinpath(output_dir, "$(tag)_gridded.csv"), "w") do rf
        hdr = "r,l"
        for d in 1:nderiv
            for (v, _) in vars_s; hdr *= ",$(v)$(suffix[d])"; end
        end
        println(rf, hdr)
        for p in 1:size(reg_pts, 1)
            row = "$(reg_pts[p, 1]),$(reg_pts[p, 2])"
            for d in 1:nderiv
                for (_, vi) in vars_s; row *= ",$(reg_phys[p, vi, d])"; end
            end
            println(rf, row)
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

# ════════════════════════════════════════════════════════════════════════════
# JLD2 binary serialization
# ════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# save_grid / load_grid
# ────────────────────────────────────────────────────────────────────────────

"""
    save_grid(filename::String, grid::SpringsteelGrid; compress::Bool=true)
    load_grid(filename::String) -> SpringsteelGrid

Save a [`SpringsteelGrid`](@ref) to a JLD2 binary file (`save_grid`) or reload
one previously written by `save_grid` (`load_grid`).

`save_grid` stores only the three serializable components—`params`, `spectral`,
and `physical`—together with a format-version tag.  FFTW plans and other
non-serializable objects are intentionally omitted.

`load_grid` reads those components, calls [`createGrid`](@ref) to reconstruct
all basis objects and FFTW plans from `params`, then copies the archived
spectral and physical arrays into the fresh grid.

# Arguments — `save_grid`
- `filename`: Destination path (conventionally `*.jld2`).
- `grid`: Any [`SpringsteelGrid`](@ref) instance.
- `compress`: When `true` (default) data are compressed with Zstd via JLD2.

# Arguments — `load_grid`
- `filename`: Path to a JLD2 file previously written by `save_grid`.

# Returns — `load_grid`
A fully functional [`SpringsteelGrid`](@ref) with the same geometry,
parameters, spectral coefficients, and physical values as the saved grid.
FFTW plans are freshly initialised and ready for use.

# Example
```julia
save_grid("output.jld2", grid)
grid2 = load_grid("output.jld2")
@assert grid2.spectral ≈ grid.spectral
```

See also: [`createGrid`](@ref), [`write_grid`](@ref), [`gridTransform!`](@ref)
"""
function save_grid(filename::String, grid::SpringsteelGrid; compress::Bool=true)
    jldopen(filename, "w"; compress=compress) do f
        f["format_version"] = "1.0"
        f["params"]         = grid.params
        f["spectral"]       = grid.spectral
        f["physical"]       = grid.physical
    end
    return nothing
end

"""
    load_grid(filename::String) -> SpringsteelGrid

Reload a grid previously written by [`save_grid`](@ref). Reads the
serialised `params`, `spectral`, and `physical` arrays from `filename`,
calls [`createGrid`](@ref) to reconstruct basis objects and FFTW plans,
then copies the archived arrays into the fresh grid.

The returned grid is fully functional — transforms, interpolation, and
the solver can all be applied to it immediately.
"""
function load_grid(filename::String)
    params, spectral, physical = jldopen(filename, "r") do f
        f["params"], f["spectral"], f["physical"]
    end
    grid = createGrid(params)
    grid.spectral .= spectral
    grid.physical .= physical
    return grid
end

# ════════════════════════════════════════════════════════════════════════════
# NetCDF output
# ════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# write_netcdf — 1D (NoBasisArray j and k)
# ────────────────────────────────────────────────────────────────────────────

"""
    write_netcdf(filename::String, grid::SpringsteelGrid{G, I, NoBasisArray, NoBasisArray};
                 include_derivatives::Bool=false,
                 global_attributes::Dict{String,Any}=Dict{String,Any}())

Write a CF-1.12-compliant NetCDF file for a 1-D [`SpringsteelGrid`](@ref) on
its regular output grid.

The file contains one dimension (`"x"` for spline/Fourier-based grids,
`"z"` for Chebyshev-based grids), one coordinate variable with the same
name, and one data variable per entry in `grid.params.vars`.  If
`include_derivatives` is `true`, two additional variables per field are
added: first derivative (`"{var}_x"` / `"{var}_z"`) and second derivative
(`"{var}_xx"` / `"{var}_zz"`).

# Arguments
- `filename`: Destination path (conventionally `*.nc`).
- `grid`: 1-D [`SpringsteelGrid`](@ref) whose spectral array is current
  (i.e. [`spectralTransform!`](@ref) has been called).
- `include_derivatives`: When `true`, write first- and second-derivative
  variables alongside the field values.
- `global_attributes`: Extra key-value pairs merged into the NetCDF global
  attributes (e.g. `Dict{String,Any}("institution" => "My Lab")`).

# Example
```julia
spectralTransform!(grid)
write_netcdf("output.nc", grid; include_derivatives=true,
             global_attributes=Dict{String,Any}("institution" => "NCAR"))
```

# Returns
`nothing`

See also: [`getRegularGridpoints`](@ref), [`regularGridTransform`](@ref),
[`write_grid`](@ref)
"""
function write_netcdf(filename::String,
                      grid::SpringsteelGrid{G, I, NoBasisArray, NoBasisArray};
                      include_derivatives::Bool=false,
                      global_attributes::Dict{String,Any}=Dict{String,Any}(),
                      coordinate_attributes::Dict{String,Dict{String,Any}}=Dict{String,Dict{String,Any}}(),
                      variable_attributes::Dict{String,Dict{String,Any}}=Dict{String,Dict{String,Any}}(),
                      time::Union{Nothing,Float64}=nothing,
                      grid_mapping::Union{Nothing,Dict{String,Any}}=nothing) where {G, I}
    reg_pts  = getRegularGridpoints(grid)
    npts     = length(reg_pts)
    reg_phys = regularGridTransform(grid, reg_pts)

    # Determine coordinate name and derivative suffixes from ibasis type
    if I <: ChebyshevBasisArray
        coord_name = "z"
        coord_long = "z coordinate"
        deriv_sfx  = ("_z", "_zz")
    else
        coord_name = "x"
        coord_long = "x coordinate"
        deriv_sfx  = ("_x", "_xx")
    end

    vars_sorted = _io_vars_sorted(grid.params)

    NCDataset(filename, "c") do ds
        # ── global attributes ─────────────────────────────────────────────
        ds.attrib["Conventions"] = "CF-1.12"
        ds.attrib["title"]       = "Springsteel.jl regular grid output"
        ds.attrib["source"]      = "Springsteel.jl"
        ds.attrib["history"]     = "Created " * Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
        for (k, v) in global_attributes
            ds.attrib[k] = v
        end

        # ── time dimension (optional) ─────────────────────────────────────
        has_time = _netcdf_add_time_dim!(ds, time)

        # ── dimension & coordinate variable ───────────────────────────────
        defDim(ds, coord_name, npts)
        coord_var = defVar(ds, coord_name, Float64, (coord_name,))
        coord_var.attrib["units"]     = "1"
        coord_var.attrib["long_name"] = coord_long
        _netcdf_apply_coord_attrs!(coord_var, coord_name, coordinate_attributes)
        coord_var[:] = reg_pts

        # ── grid mapping (optional) ───────────────────────────────────────
        _netcdf_write_grid_mapping!(ds, grid_mapping)

        # ── data variables ────────────────────────────────────────────────
        data_dims = has_time ? ("time", coord_name) : (coord_name,)
        for (var_name, var_idx) in vars_sorted
            vv = defVar(ds, var_name, Float64, data_dims,
                        deflatelevel=4, fillvalue=NaN)
            vv.attrib["long_name"] = var_name
            _netcdf_apply_var_attrs!(vv, var_name, variable_attributes)
            if has_time
                vv[1, :] = reg_phys[:, var_idx, 1]
            else
                vv[:] = reg_phys[:, var_idx, 1]
            end

            if include_derivatives
                d1 = defVar(ds, var_name * deriv_sfx[1], Float64, data_dims,
                            deflatelevel=4, fillvalue=NaN)
                d1.attrib["long_name"] = var_name * " first derivative"
                if has_time
                    d1[1, :] = reg_phys[:, var_idx, 2]
                else
                    d1[:] = reg_phys[:, var_idx, 2]
                end

                d2 = defVar(ds, var_name * deriv_sfx[2], Float64, data_dims,
                            deflatelevel=4, fillvalue=NaN)
                d2.attrib["long_name"] = var_name * " second derivative"
                if has_time
                    d2[1, :] = reg_phys[:, var_idx, 3]
                else
                    d2[:] = reg_phys[:, var_idx, 3]
                end
            end
        end
    end
    return nothing
end

# ════════════════════════════════════════════════════════════════════════════
# NetCDF helpers (shared by 2-D and 3-D write_netcdf dispatches)
# ════════════════════════════════════════════════════════════════════════════

# Write standard CF global attributes plus any user-supplied extras.
function _netcdf_global_attrs!(ds, global_attributes)
    ds.attrib["Conventions"] = "CF-1.12"
    ds.attrib["title"]       = "Springsteel.jl regular grid output"
    ds.attrib["source"]      = "Springsteel.jl"
    ds.attrib["history"]     = "Created " * Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    for (k, v) in global_attributes; ds.attrib[k] = v; end
end

# Apply user-supplied coordinate attributes to a coordinate variable.
function _netcdf_apply_coord_attrs!(coord_var, coord_name, coordinate_attributes)
    if haskey(coordinate_attributes, coord_name)
        for (ak, av) in coordinate_attributes[coord_name]
            coord_var.attrib[ak] = av
        end
    end
end

# Apply user-supplied variable attributes to a data variable.
function _netcdf_apply_var_attrs!(data_var, var_name, variable_attributes)
    if haskey(variable_attributes, var_name)
        for (ak, av) in variable_attributes[var_name]
            data_var.attrib[ak] = av
        end
    end
end

# Write a grid_mapping scalar variable if grid_mapping dict is provided.
function _netcdf_write_grid_mapping!(ds, grid_mapping)
    if grid_mapping !== nothing
        gm_name = get(grid_mapping, "variable_name", "grid_mapping")
        gm = defVar(ds, gm_name, Int32, ())
        for (k, v) in grid_mapping
            k == "variable_name" && continue
            gm.attrib[k] = v
        end
        gm[:] = Int32(0)
    end
end

# Add a leading time dimension of size 1 if time value is provided.
# Returns the updated dimension tuple and whether time was added.
function _netcdf_add_time_dim!(ds, time_val)
    if time_val !== nothing
        defDim(ds, "time", 1)
        tvar = defVar(ds, "time", Float64, ("time",))
        tvar.attrib["units"]     = "seconds since 1970-01-01T00:00:00Z"
        tvar.attrib["calendar"]  = "gregorian"
        tvar.attrib["long_name"] = "time"
        tvar[:] = [time_val]
        return true
    end
    return false
end

# (name, units, long_name, standard_name_or_nothing) for each geometry × dimension.
_netcdf_coord_info(::CartesianGeometry,   ::Val{:i}) = ("x",         "1",             "x coordinate",    nothing)
_netcdf_coord_info(::CartesianGeometry,   ::Val{:j}) = ("y",         "1",             "y coordinate",    nothing)
_netcdf_coord_info(::CartesianGeometry,   ::Val{:k}) = ("z",         "1",             "z coordinate",    nothing)

_netcdf_coord_info(::CylindricalGeometry, ::Val{:i}) = ("radius",    "1",             "radial distance", nothing)
_netcdf_coord_info(::CylindricalGeometry, ::Val{:j}) = ("azimuth",   "degrees",       "azimuth angle",   nothing)
_netcdf_coord_info(::CylindricalGeometry, ::Val{:k}) = ("height",    "1",             "height",          nothing)

_netcdf_coord_info(::SphericalGeometry,   ::Val{:i}) = ("latitude",  "degrees_north", "latitude",        "latitude")
_netcdf_coord_info(::SphericalGeometry,   ::Val{:j}) = ("longitude", "degrees_east",  "longitude",       "longitude")
_netcdf_coord_info(::SphericalGeometry,   ::Val{:k}) = ("height",    "1",             "height",          nothing)

# Derivative variable name suffixes indexed by slot (1=value, 2..7=derivatives).
_netcdf_deriv_sfx(::CartesianGeometry)   = ["", "_x",   "_xx",     "_y",   "_yy",     "_z",  "_zz"]
_netcdf_deriv_sfx(::CylindricalGeometry) = ["", "_r",   "_rr",     "_az",  "_azaz",   "_z",  "_zz"]
_netcdf_deriv_sfx(::SphericalGeometry)   = ["", "_lat", "_latlat", "_lon", "_lonlon", "_z",  "_zz"]

# j-coordinate values (in output units) for j-active and 3-D grids.
# Fourier-based j (RL, SL): evenly-spaced degrees in [0, 360).
# Spline-based j (RR, RRR): LinRange in native units.
_netcdf_j_coords(::Union{CylindricalGeometry, SphericalGeometry}, gp) =
    [360.0 * (j - 1) / gp.j_regular_out for j in 1:gp.j_regular_out]
_netcdf_j_coords(::CartesianGeometry, gp) =
    collect(LinRange(gp.jMin, gp.jMax, gp.j_regular_out))

# Reshape flat (n_i*n_j, nvars, nslots) → (n_i, n_j) for one variable+slot.
function _netcdf_reshape2d(reg_phys, var_idx, slot, n_i, n_j)
    data = zeros(n_i, n_j)
    @inbounds for i in 1:n_i, j in 1:n_j
        data[i, j] = reg_phys[(i-1)*n_j + j, var_idx, slot]
    end
    return data
end

# Reshape flat (n_i*n_j*n_k, nvars, nslots) → (n_i, n_j, n_k) for one variable+slot.
function _netcdf_reshape3d(reg_phys, var_idx, slot, n_i, n_j, n_k)
    data = zeros(n_i, n_j, n_k)
    idx = 1
    @inbounds for i in 1:n_i, j in 1:n_j, k in 1:n_k
        data[i, j, k] = reg_phys[idx, var_idx, slot]
        idx += 1
    end
    return data
end

# Define and write a single compressed NetCDF variable with a long_name attr.
function _netcdf_defvar!(ds, full_name, dims, data)
    vv = defVar(ds, full_name, Float64, dims, deflatelevel=4, fillvalue=NaN)
    vv.attrib["long_name"] = full_name
    vv[:] = data
end

# ────────────────────────────────────────────────────────────────────────────
# write_netcdf — 2D j-active (RL, RR, SL; k slot is NoBasisArray)
# ────────────────────────────────────────────────────────────────────────────

"""
    write_netcdf(filename::String,
                 grid::SpringsteelGrid{G, I, J, NoBasisArray};
                 include_derivatives::Bool=false,
                 global_attributes::Dict{String,Any}=Dict{String,Any}())

Write a CF-1.12-compliant NetCDF file for a 2-D [`SpringsteelGrid`](@ref) with
an active j-dimension (e.g. `RL_Grid`, `RR_Grid`, `SL_Grid`).

The file contains two dimensions and coordinate variables. Coordinate names,
units, and attributes follow the geometry:

| Geometry | i-dim | j-dim |
|:---------|:------|:------|
| Cartesian | `x` (units `"1"`) | `y` (units `"1"`) |
| Cylindrical | `radius` (units `"1"`) | `azimuth` (units `"degrees"`) |
| Spherical | `latitude` (units `"degrees_north"`) | `longitude` (units `"degrees_east"`) |

For spherical grids, colatitude values from [`getRegularGridpoints`](@ref) are
converted to latitude (`lat = 90 − θ·180/π`) and the latitude axis is reversed
to run south-to-north (CF convention).  Data arrays are flipped accordingly.

# Arguments
- `filename`: Destination path (conventionally `*.nc`).
- `grid`: 2-D j-active [`SpringsteelGrid`](@ref) whose spectral array is current.
- `include_derivatives`: When `true`, write all five derivative slots alongside
  field values (slots 2–5: ∂/∂i, ∂²/∂i², ∂/∂j, ∂²/∂j²).
- `global_attributes`: Extra key-value pairs merged into the NetCDF global
  attributes.

# Example
```julia
spectralTransform!(rl_grid)
write_netcdf("output.nc", rl_grid)
```

# Returns
`nothing`

See also: [`getRegularGridpoints`](@ref), [`regularGridTransform`](@ref),
[`write_netcdf`](@ref)
"""
function write_netcdf(filename::String,
                      grid::SpringsteelGrid{G, I, J, NoBasisArray};
                      include_derivatives::Bool=false,
                      global_attributes::Dict{String,Any}=Dict{String,Any}(),
                      coordinate_attributes::Dict{String,Dict{String,Any}}=Dict{String,Dict{String,Any}}(),
                      variable_attributes::Dict{String,Dict{String,Any}}=Dict{String,Dict{String,Any}}(),
                      time::Union{Nothing,Float64}=nothing,
                      grid_mapping::Union{Nothing,Dict{String,Any}}=nothing) where {
                      G <: AbstractGeometry, I,
                      J <: Union{SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}}
    gp   = grid.params
    geom = G()
    n_i  = gp.i_regular_out
    n_j  = gp.j_regular_out

    # Coordinate values in output units
    i_raw  = collect(LinRange(gp.iMin, gp.iMax, n_i))
    j_vals = _netcdf_j_coords(geom, gp)

    # For spherical grids, convert colatitude → latitude and reverse (south→north)
    i_vals = (G <: SphericalGeometry) ?
        reverse(90.0 .- i_raw .* (180.0 / π)) : i_raw

    (i_name, i_units, i_long, i_std) = _netcdf_coord_info(geom, Val(:i))
    (j_name, j_units, j_long, j_std) = _netcdf_coord_info(geom, Val(:j))

    # Evaluate on regular grid
    reg_pts  = getRegularGridpoints(grid)
    reg_phys = regularGridTransform(grid, reg_pts)   # (n_i*n_j, nvars, 5)

    vars_sorted = _io_vars_sorted(gp)
    sfx         = _netcdf_deriv_sfx(geom)            # 7-element
    n_slots     = include_derivatives ? 5 : 1

    NCDataset(filename, "c") do ds
        _netcdf_global_attrs!(ds, global_attributes)

        # ── time dimension (optional) ─────────────────────────────────────
        has_time = _netcdf_add_time_dim!(ds, time)

        defDim(ds, i_name, n_i)
        defDim(ds, j_name, n_j)

        # Coordinate variables
        cv_i = defVar(ds, i_name, Float64, (i_name,))
        cv_i.attrib["units"]     = i_units
        cv_i.attrib["long_name"] = i_long
        (i_std !== nothing) && (cv_i.attrib["standard_name"] = i_std)
        _netcdf_apply_coord_attrs!(cv_i, i_name, coordinate_attributes)
        cv_i[:] = i_vals

        cv_j = defVar(ds, j_name, Float64, (j_name,))
        cv_j.attrib["units"]     = j_units
        cv_j.attrib["long_name"] = j_long
        (j_std !== nothing) && (cv_j.attrib["standard_name"] = j_std)
        _netcdf_apply_coord_attrs!(cv_j, j_name, coordinate_attributes)
        cv_j[:] = j_vals

        # ── grid mapping (optional) ───────────────────────────────────────
        _netcdf_write_grid_mapping!(ds, grid_mapping)

        # Data variables (value + up to 4 derivative slots)
        data_dims = has_time ? ("time", i_name, j_name) : (i_name, j_name)
        for (var_name, var_idx) in vars_sorted
            for slot in 1:n_slots
                data_2d = _netcdf_reshape2d(reg_phys, var_idx, slot, n_i, n_j)
                (G <: SphericalGeometry) && (data_2d = data_2d[end:-1:1, :])
                full_name = var_name * sfx[slot]
                vv = defVar(ds, full_name, Float64, data_dims, deflatelevel=4, fillvalue=NaN)
                vv.attrib["long_name"] = full_name
                _netcdf_apply_var_attrs!(vv, full_name, variable_attributes)
                if has_time
                    vv[1, :, :] = data_2d
                else
                    vv[:] = data_2d
                end
            end
        end
    end
    return nothing
end

# ────────────────────────────────────────────────────────────────────────────
# write_netcdf — 2D k-active (RZ; j slot is NoBasisArray)
# ────────────────────────────────────────────────────────────────────────────

"""
    write_netcdf(filename::String,
                 grid::SpringsteelGrid{G, I, NoBasisArray, K};
                 include_derivatives::Bool=false,
                 global_attributes::Dict{String,Any}=Dict{String,Any}())

Write a CF-1.12-compliant NetCDF file for a 2-D [`SpringsteelGrid`](@ref) with
an active k-dimension (e.g. `RZ_Grid`).

The file contains two dimensions: the i-coordinate and the k-coordinate.
Coordinate names follow the geometry (e.g. `x` and `z` for Cartesian grids).

If `include_derivatives=true`, five slots are written per variable:
value, ∂/∂i, ∂²/∂i², ∂/∂k, ∂²/∂k² (suffixes `"_x"`, `"_xx"`, `"_z"`, `"_zz"`
for Cartesian grids).

# Arguments
- `filename`: Destination path (conventionally `*.nc`).
- `grid`: 2-D k-active [`SpringsteelGrid`](@ref) whose spectral array is current
  (i.e. [`spectralTransform!`](@ref) has been called).
- `include_derivatives`: When `true`, write first- and second-derivative
  variables alongside the field values.
- `global_attributes`: Extra key-value pairs merged into the NetCDF global
  attributes (e.g. `Dict{String,Any}("institution" => "My Lab")`).

# Returns
`nothing`

# Example
```julia
spectralTransform!(rz_grid)
write_netcdf("output.nc", rz_grid; include_derivatives=true)
```

See also: [`getRegularGridpoints`](@ref), [`regularGridTransform`](@ref),
[`write_netcdf`](@ref)
"""
function write_netcdf(filename::String,
                      grid::SpringsteelGrid{G, I, NoBasisArray, K};
                      include_derivatives::Bool=false,
                      global_attributes::Dict{String,Any}=Dict{String,Any}(),
                      coordinate_attributes::Dict{String,Dict{String,Any}}=Dict{String,Dict{String,Any}}(),
                      variable_attributes::Dict{String,Dict{String,Any}}=Dict{String,Dict{String,Any}}(),
                      time::Union{Nothing,Float64}=nothing,
                      grid_mapping::Union{Nothing,Dict{String,Any}}=nothing) where {
                      G <: AbstractGeometry, I,
                      K <: Union{SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}}
    gp   = grid.params
    geom = G()
    n_i  = gp.i_regular_out
    n_k  = gp.k_regular_out

    i_vals = collect(LinRange(gp.iMin, gp.iMax, n_i))
    k_vals = collect(LinRange(gp.kMin, gp.kMax, n_k))

    (i_name, i_units, i_long, i_std) = _netcdf_coord_info(geom, Val(:i))
    (k_name, k_units, k_long, k_std) = _netcdf_coord_info(geom, Val(:k))

    # Derivative suffixes for 2D k-active: i-dim then k-dim
    # slots: 1=value, 2=∂/∂i, 3=∂²/∂i², 4=∂/∂k, 5=∂²/∂k²
    sfx_base   = _netcdf_deriv_sfx(geom)  # 7-element
    sfx_2dk    = [sfx_base[1], sfx_base[2], sfx_base[3], sfx_base[6], sfx_base[7]]
    n_slots    = include_derivatives ? 5 : 1

    reg_pts  = getRegularGridpoints(grid)
    reg_phys = regularGridTransform(grid, reg_pts)   # (n_i*n_k, nvars, 5)

    vars_sorted = _io_vars_sorted(gp)

    NCDataset(filename, "c") do ds
        _netcdf_global_attrs!(ds, global_attributes)

        # ── time dimension (optional) ─────────────────────────────────────
        has_time = _netcdf_add_time_dim!(ds, time)

        defDim(ds, i_name, n_i)
        defDim(ds, k_name, n_k)

        cv_i = defVar(ds, i_name, Float64, (i_name,))
        cv_i.attrib["units"]     = i_units
        cv_i.attrib["long_name"] = i_long
        (i_std !== nothing) && (cv_i.attrib["standard_name"] = i_std)
        _netcdf_apply_coord_attrs!(cv_i, i_name, coordinate_attributes)
        cv_i[:] = i_vals

        cv_k = defVar(ds, k_name, Float64, (k_name,))
        cv_k.attrib["units"]     = k_units
        cv_k.attrib["long_name"] = k_long
        (k_std !== nothing) && (cv_k.attrib["standard_name"] = k_std)
        _netcdf_apply_coord_attrs!(cv_k, k_name, coordinate_attributes)
        cv_k[:] = k_vals

        # ── grid mapping (optional) ───────────────────────────────────────
        _netcdf_write_grid_mapping!(ds, grid_mapping)

        data_dims = has_time ? ("time", i_name, k_name) : (i_name, k_name)
        for (var_name, var_idx) in vars_sorted
            for slot in 1:n_slots
                data_2d   = _netcdf_reshape2d(reg_phys, var_idx, slot, n_i, n_k)
                full_name = var_name * sfx_2dk[slot]
                vv = defVar(ds, full_name, Float64, data_dims, deflatelevel=4, fillvalue=NaN)
                vv.attrib["long_name"] = full_name
                _netcdf_apply_var_attrs!(vv, full_name, variable_attributes)
                if has_time
                    vv[1, :, :] = data_2d
                else
                    vv[:] = data_2d
                end
            end
        end
    end
    return nothing
end

# ────────────────────────────────────────────────────────────────────────────
# write_netcdf — 3D j+k active (RLZ, RRR, SLZ)
# ────────────────────────────────────────────────────────────────────────────

"""
    write_netcdf(filename::String,
                 grid::SpringsteelGrid{G, I, J, K};
                 include_derivatives::Bool=false,
                 global_attributes::Dict{String,Any}=Dict{String,Any}())

Write a CF-1.12-compliant NetCDF file for a 3-D [`SpringsteelGrid`](@ref)
(e.g. `RLZ_Grid`, `RRR_Grid`, `SLZ_Grid`).

The file contains three dimensions and coordinate variables. Coordinate names,
units, and attributes follow the geometry:

| Geometry | i-dim | j-dim | k-dim |
|:---------|:------|:------|:------|
| Cartesian | `x` (`"1"`) | `y` (`"1"`) | `z` (`"1"`) |
| Cylindrical | `radius` (`"1"`) | `azimuth` (`"degrees"`) | `height` (`"1"`) |
| Spherical | `latitude` (`"degrees_north"`) | `longitude` (`"degrees_east"`) | `height` (`"1"`) |

For spherical grids, colatitude → latitude conversion and south-to-north
latitude axis reversal are applied (CF convention).

If `include_derivatives=true`, seven slots are written per variable:
value plus the six first/second derivative pairs for each dimension.

# Arguments
- `filename`: Destination path (conventionally `*.nc`).
- `grid`: 3-D [`SpringsteelGrid`](@ref) whose spectral array is current
  (i.e. [`spectralTransform!`](@ref) has been called).
- `include_derivatives`: When `true`, write first- and second-derivative
  variables alongside the field values.
- `global_attributes`: Extra key-value pairs merged into the NetCDF global
  attributes (e.g. `Dict{String,Any}("institution" => "My Lab")`).

# Returns
`nothing`

# Example
```julia
spectralTransform!(rlz_grid)
write_netcdf("output.nc", rlz_grid; include_derivatives=true)
```

See also: [`getRegularGridpoints`](@ref), [`regularGridTransform`](@ref),
[`write_netcdf`](@ref)
"""
function write_netcdf(filename::String,
                      grid::SpringsteelGrid{G, I, J, K};
                      include_derivatives::Bool=false,
                      global_attributes::Dict{String,Any}=Dict{String,Any}(),
                      coordinate_attributes::Dict{String,Dict{String,Any}}=Dict{String,Dict{String,Any}}(),
                      variable_attributes::Dict{String,Dict{String,Any}}=Dict{String,Dict{String,Any}}(),
                      time::Union{Nothing,Float64}=nothing,
                      grid_mapping::Union{Nothing,Dict{String,Any}}=nothing) where {
                      G <: AbstractGeometry, I,
                      J <: Union{SplineBasisArray, FourierBasisArray, ChebyshevBasisArray},
                      K <: Union{SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}}
    gp   = grid.params
    geom = G()
    n_i  = gp.i_regular_out
    n_j  = gp.j_regular_out
    n_k  = gp.k_regular_out

    # Coordinate values in output units
    i_raw  = collect(LinRange(gp.iMin, gp.iMax, n_i))
    j_vals = _netcdf_j_coords(geom, gp)
    k_vals = collect(LinRange(gp.kMin, gp.kMax, n_k))

    i_vals = (G <: SphericalGeometry) ?
        reverse(90.0 .- i_raw .* (180.0 / π)) : i_raw

    (i_name, i_units, i_long, i_std) = _netcdf_coord_info(geom, Val(:i))
    (j_name, j_units, j_long, j_std) = _netcdf_coord_info(geom, Val(:j))
    (k_name, k_units, k_long, k_std) = _netcdf_coord_info(geom, Val(:k))

    reg_pts  = getRegularGridpoints(grid)
    reg_phys = regularGridTransform(grid, reg_pts)   # (n_i*n_j*n_k, nvars, 7)

    vars_sorted = _io_vars_sorted(gp)
    sfx         = _netcdf_deriv_sfx(geom)            # 7-element
    n_slots     = include_derivatives ? 7 : 1

    NCDataset(filename, "c") do ds
        _netcdf_global_attrs!(ds, global_attributes)

        # ── time dimension (optional) ─────────────────────────────────────
        has_time = _netcdf_add_time_dim!(ds, time)

        defDim(ds, i_name, n_i)
        defDim(ds, j_name, n_j)
        defDim(ds, k_name, n_k)

        cv_i = defVar(ds, i_name, Float64, (i_name,))
        cv_i.attrib["units"]     = i_units
        cv_i.attrib["long_name"] = i_long
        (i_std !== nothing) && (cv_i.attrib["standard_name"] = i_std)
        _netcdf_apply_coord_attrs!(cv_i, i_name, coordinate_attributes)
        cv_i[:] = i_vals

        cv_j = defVar(ds, j_name, Float64, (j_name,))
        cv_j.attrib["units"]     = j_units
        cv_j.attrib["long_name"] = j_long
        (j_std !== nothing) && (cv_j.attrib["standard_name"] = j_std)
        _netcdf_apply_coord_attrs!(cv_j, j_name, coordinate_attributes)
        cv_j[:] = j_vals

        cv_k = defVar(ds, k_name, Float64, (k_name,))
        cv_k.attrib["units"]     = k_units
        cv_k.attrib["long_name"] = k_long
        (k_std !== nothing) && (cv_k.attrib["standard_name"] = k_std)
        _netcdf_apply_coord_attrs!(cv_k, k_name, coordinate_attributes)
        cv_k[:] = k_vals

        # ── grid mapping (optional) ───────────────────────────────────────
        _netcdf_write_grid_mapping!(ds, grid_mapping)

        data_dims = has_time ? ("time", i_name, j_name, k_name) : (i_name, j_name, k_name)
        for (var_name, var_idx) in vars_sorted
            for slot in 1:n_slots
                data_3d = _netcdf_reshape3d(reg_phys, var_idx, slot, n_i, n_j, n_k)
                (G <: SphericalGeometry) && (data_3d = data_3d[end:-1:1, :, :])
                full_name = var_name * sfx[slot]
                vv = defVar(ds, full_name, Float64, data_dims, deflatelevel=4, fillvalue=NaN)
                vv.attrib["long_name"] = full_name
                _netcdf_apply_var_attrs!(vv, full_name, variable_attributes)
                if has_time
                    vv[1, :, :, :] = data_3d
                else
                    vv[:] = data_3d
                end
            end
        end
    end
    return nothing
end

# ════════════════════════════════════════════════════════════════════════════
# NetCDF input
# ════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# read_netcdf
# ────────────────────────────────────────────────────────────────────────────

"""
    read_netcdf(filename::String) -> Dict{String, Any}

Read a CF-compliant NetCDF file (typically produced by [`write_netcdf`](@ref))
and return its contents as a plain [`Dict`].

The returned dictionary has four string-keyed entries:

| Key            | Type                          | Contents                          |
|:-------------- |:----------------------------- |:--------------------------------- |
| `"dimensions"` | `Dict{String, Int}`           | dimension name → length           |
| `"coordinates"`| `Dict{String, Vector{Float64}}`| coordinate name → values         |
| `"variables"`  | `Dict{String, Array{Float64}}`| data variable name → array        |
| `"attributes"` | `Dict{String, Any}`           | global attribute name → value     |

Coordinate variables are identified by sharing their name with a NetCDF
dimension.  All other variables are returned under `"variables"`.  This
format-independent structure can be passed directly to plotting libraries
or used for further analysis without requiring a [`SpringsteelGrid`](@ref).

# Arguments
- `filename`: Path to an existing NetCDF file.

# Returns
`Dict{String, Any}` as described above.

# Example
```julia
spectralTransform!(grid)
write_netcdf("output.nc", grid)
data = read_netcdf("output.nc")
println(data["attributes"]["Conventions"])  # "CF-1.12"
x    = data["coordinates"]["x"]
u    = data["variables"]["u"]
```

See also: [`write_netcdf`](@ref), [`read_physical_grid`](@ref)
"""
function read_netcdf(filename::String)
    result = Dict{String, Any}()

    NCDataset(filename, "r") do ds
        # Dimensions
        dims = Dict{String, Int}()
        for (name, dim) in ds.dim
            dims[name] = dim   # ds.dim iterates as (name, Int) pairs
        end
        result["dimensions"] = dims

        # Coordinates (variables that share their dimension name)
        coords = Dict{String, Vector{Float64}}()
        for name in keys(dims)
            if haskey(ds, name)
                coords[name] = Array(ds[name])
            end
        end
        result["coordinates"] = coords

        # Variables (everything that's not a coordinate)
        vars = Dict{String, Array{Float64}}()
        for name in keys(ds)
            if !haskey(dims, name)
                vars[name] = Array(ds[name])
            end
        end
        result["variables"] = vars

        # Global attributes
        attrs = Dict{String, Any}()
        for name in keys(ds.attrib)
            attrs[name] = ds.attrib[name]
        end
        result["attributes"] = attrs
    end

    return result
end

