# deprecated.jl — Backward-compatibility helpers
#
# Provides:
#   • convert_to_springsteel_params(gp::GridParameters) -> SpringsteelGridParameters
#
# The legacy createGrid(::GridParameters) method that accepts old GridParameters structs
# delegates to the unified SpringsteelGridParameters path via convert_to_springsteel_params.

"""
    convert_to_springsteel_params(gp::GridParameters) -> SpringsteelGridParameters

Convert a legacy [`GridParameters`](@ref) instance to the equivalent
[`SpringsteelGridParameters`](@ref), mapping old field names (`xmin`, `rDim`, `b_rDim`,
`lDim`, `b_lDim`, `zmin`, `zmax`, `zDim`, `b_zDim`, `kmax`, …) to new names
(`iMin`, `iDim`, `b_iDim`, `jDim`, `b_jDim`, `kMin`, `kMax`, `kDim`, `b_kDim`,
`max_wavenumber`, …).

The geometry string is passed through unchanged; `"R"`, `"RL"`, `"RZ"`, `"RR"`, `"RLZ"`,
`"RRR"` are all valid inputs.

# Field mapping

| Old (`GridParameters`) | New (`SpringsteelGridParameters`) |
|:---------------------- |:--------------------------------- |
| `xmin`, `xmax`         | `iMin`, `iMax`                    |
| `rDim`, `b_rDim`       | `iDim`, `b_iDim`                  |
| `ymin`, `ymax`         | `jMin`, `jMax`                    |
| `lDim`, `b_lDim`       | `jDim`, `b_jDim`                  |
| `kmax`                 | `max_wavenumber`                  |
| `zmin`, `zmax`         | `kMin`, `kMax`                    |
| `zDim`, `b_zDim`       | `kDim`, `b_kDim`                  |
| `BCU`, `BCD`           | `BCU`, `BCD` (unchanged)          |
| `BCB`, `BCT`           | `BCB`, `BCT` (unchanged)          |

# Example
```julia
gp_old = GridParameters(geometry = "RL", xmin=0.0, xmax=100.0, num_cells=10,
    BCL = Dict("u" => CubicBSpline.R0), BCR = Dict("u" => CubicBSpline.R0),
    vars = Dict("u" => 1))

gp_new = convert_to_springsteel_params(gp_old)
grid   = createGrid(gp_new)   # produces SpringsteelGrid{CylindricalGeometry, ...}
```

See also: [`SpringsteelGridParameters`](@ref), [`GridParameters`](@ref),
[`createGrid`](@ref)
"""
function convert_to_springsteel_params(gp::GridParameters)
    return SpringsteelGridParameters(
        geometry       = gp.geometry,
        iMin           = gp.xmin,
        iMax           = gp.xmax,
        num_cells      = gp.num_cells,
        iDim           = gp.rDim,
        b_iDim         = gp.b_rDim,
        l_q            = gp.l_q,
        BCL            = gp.BCL,
        BCR            = gp.BCR,
        jMin           = gp.ymin,
        jMax           = gp.ymax,
        max_wavenumber = gp.kmax,
        jDim           = gp.lDim,
        b_jDim         = gp.b_lDim,
        BCU            = gp.BCU,
        BCD            = gp.BCD,
        kMin           = gp.zmin,
        kMax           = gp.zmax,
        kDim           = gp.zDim,
        b_kDim         = gp.b_zDim,
        BCB            = gp.BCB,
        BCT            = gp.BCT,
        vars           = gp.vars,
        fourier_filter = gp.fourier_filter,
        chebyshev_filter = gp.chebyshev_filter,
        spectralIndexL = gp.spectralIndexL,
        spectralIndexR = gp.spectralIndexR,
        patchOffsetL   = gp.patchOffsetL,
        patchOffsetR   = gp.patchOffsetR,
        tile_num       = gp.tile_num)
end
"""
    createGrid(gp::GridParameters) -> SpringsteelGrid

    createGrid(gp::GridParameters)

Legacy entry point for grid creation using the old [`GridParameters`](@ref) struct.
Converts `gp` to a [`SpringsteelGridParameters`](@ref) via
[`convert_to_springsteel_params`](@ref) and delegates to the unified
[`createGrid(::SpringsteelGridParameters)`](@ref) factory.

# Arguments
- `gp::GridParameters`: Legacy parameter struct

# Returns
A [`SpringsteelGrid`](@ref) instance (same type as `createGrid(::SpringsteelGridParameters)`).

# Deprecation
Use [`SpringsteelGridParameters`](@ref) directly for new code.

# Example
```julia
# Legacy call (still works):
gp = GridParameters(geometry="R", xmin=0.0, xmax=10.0, num_cells=10,
    BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
    vars=Dict("u"=>1))
grid = createGrid(gp)  # returns SpringsteelGrid{CartesianGeometry, SplineBasisArray, ...}
```

See also: [`SpringsteelGridParameters`](@ref), [`convert_to_springsteel_params`](@ref)
"""
function createGrid(gp::GridParameters)
    return createGrid(convert_to_springsteel_params(gp))
end