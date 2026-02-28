# basis_interface.jl — Generic accessors for Spline1D, Fourier1D, Chebyshev1D
#
# Provides three accessor functions with dispatch over all three basis types:
#   • gridpoints(obj)    — physical grid-point locations
#   • spectral_dim(obj)  — number of spectral coefficients
#   • physical_dim(obj)  — number of physical gridpoints
#
# These allow algorithms operating on SpringsteelGrid to interrogate basis
# objects without caring about which concrete type they hold.

# ────────────────────────────────────────────────────────────────────────────
# gridpoints
# ────────────────────────────────────────────────────────────────────────────

"""
    gridpoints(obj) -> Vector{Float64}

    gridpoints(obj::CubicBSpline.Spline1D)   -> obj.mishPoints
    gridpoints(obj::Fourier.Fourier1D)        -> obj.mishPoints
    gridpoints(obj::Chebyshev.Chebyshev1D)   -> obj.mishPoints

Return the physical grid-point locations stored in the basis object `obj`.

All three basis types store this information in the field `mishPoints`, but the
accessor ensures forward-compatible, type-stable access regardless of internal
layout changes.

# Arguments
- `obj`: a `Spline1D`, `Fourier1D`, or `Chebyshev1D` basis object.

# Returns
A `Vector{Float64}` of length `physical_dim(obj)` containing the coordinates
of the evaluation points in physical space.

# Example
```julia
pts = gridpoints(grid.ibasis.data[var])   # radial gridpoints for variable var
```

See also: [`spectral_dim`](@ref), [`physical_dim`](@ref)
"""
gridpoints(obj::CubicBSpline.Spline1D)  = obj.mishPoints
gridpoints(obj::Fourier.Fourier1D)      = obj.mishPoints
gridpoints(obj::Chebyshev.Chebyshev1D) = obj.mishPoints

# ────────────────────────────────────────────────────────────────────────────
# spectral_dim
# ────────────────────────────────────────────────────────────────────────────

"""
    spectral_dim(obj) -> Int

    spectral_dim(obj::CubicBSpline.Spline1D)   -> obj.params.bDim
    spectral_dim(obj::Fourier.Fourier1D)        -> obj.params.bDim
    spectral_dim(obj::Chebyshev.Chebyshev1D)   -> obj.params.bDim

Return the number of spectral (basis) coefficients associated with `obj`.

For `Spline1D`, `bDim = num_cells + 3` is stored as a direct field.
For `Fourier1D` and `Chebyshev1D`, the value lives in the nested `params`
struct; this accessor hides that distinction.

# Arguments
- `obj`: a `Spline1D`, `Fourier1D`, or `Chebyshev1D` basis object.

# Returns
An `Int` equal to the length of the spectral coefficient vector for this object.

# Example
```julia
nb = spectral_dim(grid.ibasis.data[var])   # number of spline coefficients
```

See also: [`physical_dim`](@ref), [`gridpoints`](@ref)
"""
spectral_dim(obj::CubicBSpline.Spline1D)  = obj.params.bDim   # via params
spectral_dim(obj::Fourier.Fourier1D)      = obj.params.bDim   # via params
spectral_dim(obj::Chebyshev.Chebyshev1D) = obj.params.bDim   # via params

# ────────────────────────────────────────────────────────────────────────────
# physical_dim
# ────────────────────────────────────────────────────────────────────────────

"""
    physical_dim(obj) -> Int

    physical_dim(obj::CubicBSpline.Spline1D)   -> obj.params.mishDim
    physical_dim(obj::Fourier.Fourier1D)        -> obj.params.yDim
    physical_dim(obj::Chebyshev.Chebyshev1D)   -> obj.params.zDim

Return the number of physical gridpoints for the basis object `obj`.

For `Spline1D`, `mishDim = num_cells * CubicBSpline.mubar` is a direct field.
For `Fourier1D` and `Chebyshev1D`, the values are `yDim` and `zDim`
respectively inside the nested `params` struct.

# Arguments
- `obj`: a `Spline1D`, `Fourier1D`, or `Chebyshev1D` basis object.

# Returns
An `Int` equal to the number of physical-space evaluation points.

# Example
```julia
np = physical_dim(grid.ibasis.data[var])   # number of radial gridpoints
```

See also: [`spectral_dim`](@ref), [`gridpoints`](@ref)
"""
physical_dim(obj::CubicBSpline.Spline1D)  = obj.params.mishDim  # via params
physical_dim(obj::Fourier.Fourier1D)      = obj.params.yDim     # via params
physical_dim(obj::Chebyshev.Chebyshev1D) = obj.params.zDim     # via params
