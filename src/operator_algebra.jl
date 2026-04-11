# ────────────────────────────────────────────────────────────────────────────
# operator_algebra.jl — object-level DSL for building PDE operators
# ────────────────────────────────────────────────────────────────────────────
#
# Lets users write differential operators the way the math reads:
#
#     L = ∂ᵢ^2 + ∂ⱼ^2                 # generic axes
#     L = ∂_r^2 + ∂_z^2               # physical names (resolved per-geometry)
#     L = α*∂ᵢ^2 + β*∂ᵢ + γ           # spatially varying / scalar coefficients
#
# The AST built by these expressions lowers to the existing `Vector{OperatorTerm}`
# format consumed by `assemble_operator` (see solver.jl). No changes to the
# low-level solver path — this is strictly an additive front end for S1 of the
# solver refactor (see agent_files/plan_solver_refactor.md).
#
# Design: three AST node types.
#   DerivMono   — a product of per-axis derivative orders, e.g. ∂ᵢ² ∂ⱼ
#   ScaledMono  — a DerivMono scaled by a coefficient (Float/Nothing/Vector/…)
#   OperatorExpr — a sum of ScaledMonos (the thing users actually hold)
#
# Operator overloading builds these incrementally. `_lower(expr, grid, var)`
# walks the tree and emits `Vector{OperatorTerm}`.

"""
    DerivMono

A product of per-axis derivative orders, e.g. `∂ᵢ²` or `∂ᵢ ∂ⱼ`.

Axes are stored as `Symbol → Int` pairs. Symbols may be generic (`:i`, `:j`,
`:k`) or physical (`:r`, `:z`, `:θ`, `:λ`, `:x`, `:y`); physical names are
translated to the corresponding generic axis at lowering time based on the
grid's geometry, and an error is raised if the mapping is invalid.
"""
struct DerivMono
    orders::Dict{Symbol, Int}
end

DerivMono(pairs::Pair{Symbol,Int}...) = DerivMono(Dict(pairs...))
const _IDENTITY_MONO = DerivMono(Dict{Symbol,Int}())   # no derivatives = identity

"""
    ScaledMono

A single term in an operator expression: `coefficient * DerivMono`.

The coefficient may be:
- `Nothing` (treated as `1.0` at lowering)
- `Float64` / `Int` (scalar)
- `Vector{Float64}` (spatially varying, pre-built)

S5 of the solver refactor will extend this to accept `Function` and `Symbol`
(grid-variable lookup) via `_resolve_coeff`.
"""
struct ScaledMono
    coeff::Any
    mono::DerivMono
end

"""
    OperatorExpr

Sum of `ScaledMono`s. This is the object users hold after combining
derivative atoms with `+`, `-`, `*`, `^`. Lowers to `Vector{OperatorTerm}`
via `_lower`.
"""
struct OperatorExpr
    terms::Vector{ScaledMono}
end

# Promote everything to OperatorExpr for uniform algebra.
_as_expr(m::DerivMono)    = OperatorExpr([ScaledMono(nothing, m)])
_as_expr(s::ScaledMono)   = OperatorExpr([s])
_as_expr(e::OperatorExpr) = e

# ────────────────────────────────────────────────────────────────────────────
# Derivative atoms — generic and physical aliases
# ────────────────────────────────────────────────────────────────────────────

# Generic axes (what the low-level OperatorTerm uses directly)
const ∂ᵢ = DerivMono(:i => 1)
const ∂ⱼ = DerivMono(:j => 1)
const ∂ₖ = DerivMono(:k => 1)
const d_i = ∂ᵢ
const d_j = ∂ⱼ
const d_k = ∂ₖ

# Physical names — resolved against grid.params.geometry at lowering time.
# The mapping is:
#   Cartesian   : x → i, y → j, z → k
#   Cylindrical : r → i, λ → j, z → k     (θ is an alias for λ)
#   Spherical   : θ → i, λ → j, z → k     (z is the radial coordinate here)
# Any physical alias used on a geometry that doesn't define it raises an error.
const ∂_x = DerivMono(:x => 1)
const ∂_y = DerivMono(:y => 1)
const ∂_z = DerivMono(:z => 1)
const ∂_r = DerivMono(:r => 1)
const ∂_θ = DerivMono(:θ => 1)
const ∂_λ = DerivMono(:λ => 1)

const d_x = ∂_x
const d_y = ∂_y
const d_z = ∂_z
const d_r = ∂_r
const d_theta = ∂_θ
const d_lambda = ∂_λ

# ────────────────────────────────────────────────────────────────────────────
# Operator overloads — build the AST
# ────────────────────────────────────────────────────────────────────────────

# Powers: (∂ᵢ)^p means each axis order × p
function Base.:^(m::DerivMono, p::Integer)
    p < 0 && throw(ArgumentError("DerivMono power must be ≥ 0, got $p"))
    p == 0 && return _IDENTITY_MONO
    return DerivMono(Dict(k => v * Int(p) for (k, v) in m.orders))
end

# Product of two DerivMonos adds orders per axis.
function Base.:*(a::DerivMono, b::DerivMono)
    orders = Dict{Symbol, Int}()
    for (k, v) in a.orders; orders[k] = v; end
    for (k, v) in b.orders; orders[k] = get(orders, k, 0) + v; end
    return DerivMono(orders)
end

# Scalar / vector coefficient times DerivMono → ScaledMono → OperatorExpr
_is_coeff(::Number) = true
_is_coeff(::Vector{Float64}) = true
_is_coeff(::Nothing) = true
_is_coeff(::Any) = false

Base.:*(c, m::DerivMono) = _is_coeff(c) ?
    OperatorExpr([ScaledMono(c === nothing ? nothing : Float64(c), m)]) :
    error("Coefficient must be Number, Vector{Float64}, or Nothing")
Base.:*(c::Vector{Float64}, m::DerivMono) = OperatorExpr([ScaledMono(c, m)])
Base.:*(m::DerivMono, c) = c * m

# Scalar times OperatorExpr distributes
function Base.:*(c, e::OperatorExpr)
    _is_coeff(c) || error("Coefficient must be Number, Vector{Float64}, or Nothing")
    cf = c isa Number ? Float64(c) : c
    return OperatorExpr([ScaledMono(_combine_coeff(cf, s.coeff), s.mono) for s in e.terms])
end
Base.:*(c::Vector{Float64}, e::OperatorExpr) =
    OperatorExpr([ScaledMono(_combine_coeff(c, s.coeff), s.mono) for s in e.terms])
Base.:*(e::OperatorExpr, c) = c * e

# Combine an outer coefficient with an inner ScaledMono coefficient during
# distribution. Stays within the supported types.
_combine_coeff(outer::Nothing, inner) = inner
_combine_coeff(outer, inner::Nothing) = outer
_combine_coeff(outer::Float64, inner::Float64) = outer * inner
_combine_coeff(outer::Float64, inner::Vector{Float64}) = outer .* inner
_combine_coeff(outer::Vector{Float64}, inner::Float64) = outer .* inner
_combine_coeff(outer::Vector{Float64}, inner::Vector{Float64}) = outer .* inner

# Sums — promote both sides to OperatorExpr then concatenate terms.
Base.:+(a::Union{DerivMono,ScaledMono,OperatorExpr},
        b::Union{DerivMono,ScaledMono,OperatorExpr}) =
    OperatorExpr(vcat(_as_expr(a).terms, _as_expr(b).terms))

# Unary minus and binary minus
Base.:-(m::DerivMono) = OperatorExpr([ScaledMono(-1.0, m)])
Base.:-(e::OperatorExpr) = -1.0 * e
Base.:-(a::Union{DerivMono,ScaledMono,OperatorExpr},
        b::Union{DerivMono,ScaledMono,OperatorExpr}) =
    a + (-1.0 * _as_expr(b))

# Scaled-mono power (rare but supported): (α*∂ᵢ)^2 = α² * ∂ᵢ²
function Base.:^(e::OperatorExpr, p::Integer)
    p < 0 && throw(ArgumentError("OperatorExpr power must be ≥ 0, got $p"))
    p == 0 && return _as_expr(_IDENTITY_MONO)
    p == 1 && return e
    length(e.terms) == 1 || throw(ArgumentError(
        "Power of a multi-term OperatorExpr is not supported (got $(length(e.terms)) terms)"))
    s = e.terms[1]
    new_coeff = s.coeff === nothing ? nothing :
                (s.coeff isa Vector{Float64} ? s.coeff .^ p : s.coeff ^ p)
    return OperatorExpr([ScaledMono(new_coeff, s.mono ^ p)])
end

# ────────────────────────────────────────────────────────────────────────────
# Lowering — OperatorExpr → Vector{OperatorTerm}
# ────────────────────────────────────────────────────────────────────────────

"""
    _geometry_axis_map(::AbstractGeometry) -> Dict{Symbol, Symbol}

Physical-axis name → generic axis symbol translation. Unknown physical names
for a given geometry error at `_lower` time with a helpful message.
"""
_geometry_axis_map(::CartesianGeometry) = Dict(:x => :i, :y => :j, :z => :k)
_geometry_axis_map(::CylindricalGeometry) =
    Dict(:r => :i, :λ => :j, :θ => :j, :z => :k)
_geometry_axis_map(::SphericalGeometry) =
    Dict(:θ => :i, :λ => :j, :z => :k)

"""
    _resolve_axis(axis, grid) -> Symbol

Translate a (possibly physical) axis symbol to the generic `:i`/`:j`/`:k` axis
name the low-level solver uses. Errors if `axis` is not a valid axis name for
the grid's geometry.
"""
function _resolve_axis(axis::Symbol, grid)
    if axis === :i || axis === :j || axis === :k
        return axis
    end
    geom = _geometry_instance(grid)
    map  = _geometry_axis_map(geom)
    haskey(map, axis) || throw(ArgumentError(
        "Physical axis `:$axis` is not defined for $(typeof(geom)) grids. " *
        "Valid physical axes for this geometry: $(collect(keys(map)))"))
    return map[axis]
end

# Extract the geometry instance from a grid (works for any SpringsteelGrid{G,...}).
_geometry_instance(grid::SpringsteelGrid{G}) where {G} = G()

"""
    _lower(expr, grid) -> Vector{OperatorTerm}

Walk an `OperatorExpr` (or `DerivMono` / `ScaledMono`) and emit the
`Vector{OperatorTerm}` format consumed by `assemble_operator`. Physical axis
names are resolved against the grid's geometry. Terms whose derivative order
exceeds 2 on any axis raise an error (the low-level solver only supports up
to second-order per axis).
"""
_lower(m::DerivMono, grid) = _lower(_as_expr(m), grid)
_lower(s::ScaledMono, grid) = _lower(_as_expr(s), grid)

function _lower(expr::OperatorExpr, grid)
    terms = OperatorTerm[]
    for s in expr.terms
        i_order, j_order, k_order = 0, 0, 0
        for (axis, ord) in s.mono.orders
            g_axis = _resolve_axis(axis, grid)
            ord > 2 && throw(ArgumentError(
                "Derivative order $ord on axis `:$axis` exceeds maximum (2)"))
            if g_axis === :i
                i_order += ord
            elseif g_axis === :j
                j_order += ord
            else  # :k
                k_order += ord
            end
        end
        (i_order > 2 || j_order > 2 || k_order > 2) && throw(ArgumentError(
            "Combined derivative order exceeds 2 on some axis " *
            "(i=$i_order, j=$j_order, k=$k_order)"))
        push!(terms, OperatorTerm(i_order, j_order, k_order, s.coeff))
    end
    return terms
end

# ────────────────────────────────────────────────────────────────────────────
# Display
# ────────────────────────────────────────────────────────────────────────────

function Base.show(io::IO, m::DerivMono)
    if isempty(m.orders)
        print(io, "I")
        return
    end
    parts = String[]
    for (ax, ord) in m.orders
        sym = ax === :i ? "∂ᵢ" : ax === :j ? "∂ⱼ" : ax === :k ? "∂ₖ" : "∂_$ax"
        push!(parts, ord == 1 ? sym : "$(sym)^$ord")
    end
    print(io, join(parts, " "))
end

function Base.show(io::IO, s::ScaledMono)
    if s.coeff === nothing
        show(io, s.mono)
    elseif s.coeff isa Number
        print(io, s.coeff, "·")
        show(io, s.mono)
    else
        print(io, "c·")
        show(io, s.mono)
    end
end

function Base.show(io::IO, e::OperatorExpr)
    isempty(e.terms) && (print(io, "0"); return)
    for (i, s) in enumerate(e.terms)
        i > 1 && print(io, " + ")
        show(io, s)
    end
end
