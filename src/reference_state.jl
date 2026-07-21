# Reference (base) state types and physical-density hydrostatic builders.
#
# The reference state is a hydrostatically balanced vertical base state whose primary
# job is to reduce computational gradients in the prognostic equations. It is shared by
# Springsteel-grid clients: Scythe consumes it for its dynamics, and Daisho will consume
# it for dual-Doppler mass continuity and hydrometeor classification.
#
# State is stored in *physical* densities (dry-air, vapor, and condensate partial
# densities) and moist entropy — not the model-specific prognostic transforms (`xi`,
# `mu`). Each concrete type carries only the profiles it needs, so dry / vapor-only /
# condensate-bearing runs do not allocate arrays of zeros. Profiles are stored as
# `(nlevels, 3)` arrays holding value, first, and second vertical derivative.

using .Thermodynamics

# ── Type hierarchy ─────────────────────────────────────────────────────────────

"""
    AbstractReferenceState

Supertype for hydrostatic reference states. Concrete subtypes
([`DryReferenceState`](@ref), [`MoistReferenceState`](@ref),
[`CondensateReferenceState`](@ref)) are consumed through a small accessor interface
(`ref_entropy`, `ref_rho_d`, `ref_rho_v`, `ref_rho_c`, `ref_sat`, `sound_speed_sq`)
so clients dispatch on the supertype and new variants (e.g. an energy-based state) can
be added without changing the interface.
"""
abstract type AbstractReferenceState end

"""
    DryReferenceState(sbar, sigmabar, rho_dbar, sound_speed_sq)

Dry hydrostatic reference: moist entropy, entropy density (σ̂ = ρ̂_d·ŝ), and dry-air density.
"""
struct DryReferenceState <: AbstractReferenceState
    sbar::Matrix{Float64}        # moist entropy (q_v = 0) [J/(kg K)]
    sigmabar::Matrix{Float64}    # entropy density rho_d*s [J/(K m^3)]
    rho_dbar::Matrix{Float64}    # dry-air density [kg/m^3]
    sound_speed_sq::Float64      # domain-mean speed of sound squared [m^2/s^2]
end

"""
    MoistReferenceState(sbar, sigmabar, rho_dbar, rho_vbar, satbar, sound_speed_sq)

Moist (vapor-bearing, condensate-free) hydrostatic reference: adds the vapor partial
density `rho_vbar` and the saturation-ratio profile `satbar` (`q_v / q_sat`, physical).
"""
struct MoistReferenceState <: AbstractReferenceState
    sbar::Matrix{Float64}
    sigmabar::Matrix{Float64}    # entropy density rho_d*s [J/(K m^3)]
    rho_dbar::Matrix{Float64}
    rho_vbar::Matrix{Float64}    # vapor partial density [kg/m^3]
    satbar::Matrix{Float64}      # saturation ratio q_v/q_sat
    sound_speed_sq::Float64
end

"""
    CondensateReferenceState(sbar, sigmabar, rho_dbar, rho_vbar, rho_cbar, satbar, sound_speed_sq)

Condensate-bearing hydrostatic reference: adds the condensate partial density
`rho_cbar`. Used for idealized saturated base states (e.g. the Bryan & Fritsch 2002
benchmark) where including the base cloud makes the hydrostatic state truly neutrally
buoyant and the prognostic perturbation zero outside the disturbance.
"""
struct CondensateReferenceState <: AbstractReferenceState
    sbar::Matrix{Float64}
    sigmabar::Matrix{Float64}    # entropy density rho_d*s [J/(K m^3)]
    rho_dbar::Matrix{Float64}
    rho_vbar::Matrix{Float64}
    rho_cbar::Matrix{Float64}    # condensate partial density [kg/m^3]
    satbar::Matrix{Float64}
    sound_speed_sq::Float64
end

"""
    PressureReferenceState(pbar, rho_dbar, rho_vbar, rho_cbar, rho_tbar, Tbar,
                           E_tbar, Q_ssbar, sound_speed_sq)

Pressure-based hydrostatic reference for total-energy equation sets: pressure [Pa],
partial and total densities, and the derived temperature, total-energy density
E_t = ρ_d e_i + ρ_t g z (at rest, Bryan & Fritsch 2002 internal energy), and
supersaturation density Q_ss = ρ_v − ρ_v*(T, p). Carries no entropy profile;
hydrostatic balance is the direct dp/dz = −ρ_t g rather than the entropy/log-density
(P_s, P_xi, P_qv) form.
"""
struct PressureReferenceState <: AbstractReferenceState
    pbar::Matrix{Float64}        # total pressure [Pa]
    rho_dbar::Matrix{Float64}    # dry-air density [kg/m^3]
    rho_vbar::Matrix{Float64}    # vapor partial density [kg/m^3]
    rho_cbar::Matrix{Float64}    # condensate partial density [kg/m^3]
    rho_tbar::Matrix{Float64}    # total density, fitted on the reference basis [kg/m^3]
    Tbar::Matrix{Float64}        # EOS temperature [K]
    E_tbar::Matrix{Float64}      # total energy density at rest [J/m^3]
    Q_ssbar::Matrix{Float64}     # supersaturation density [kg/m^3]
    sound_speed_sq::Float64      # mean gamma*p/rho_t [m^2/s^2]
end

# ── Accessor interface ─────────────────────────────────────────────────────────
# Present-profile accessors return the `(nlevels, 3)` array; accessors for a profile a
# given concrete type does not carry return the scalar `0.0` (broadcasts as a zero
# field). Not exported: clients `import Springsteel: ref_rho_d` etc. to use or extend
# them, avoiding clashes with same-named bindings during migration.

"""Moist entropy reference profile `(nlevels, 3)` [J/(kg K)]."""
ref_entropy(rs::AbstractReferenceState) = rs.sbar

"""Dry-air density reference profile `(nlevels, 3)` [kg/m^3]."""
ref_rho_d(rs::AbstractReferenceState) = rs.rho_dbar

"""Entropy-density reference profile `(nlevels, 3)` [J/(K m^3)], σ̂ = ρ̂_d·ŝ with the vertical
derivative computed on the reference basis (spectrally consistent — do NOT product-rule it)."""
ref_sigma(rs::AbstractReferenceState) = rs.sigmabar

"""Domain-mean speed of sound squared [m^2/s^2]."""
sound_speed_sq(rs::AbstractReferenceState) = rs.sound_speed_sq

"""Vapor partial-density reference profile `(nlevels, 3)` [kg/m^3]; `0.0` if absent."""
ref_rho_v(rs::DryReferenceState) = 0.0
ref_rho_v(rs::MoistReferenceState) = rs.rho_vbar
ref_rho_v(rs::CondensateReferenceState) = rs.rho_vbar

"""Condensate partial-density reference profile `(nlevels, 3)` [kg/m^3]; `0.0` if absent."""
ref_rho_c(rs::DryReferenceState) = 0.0
ref_rho_c(rs::MoistReferenceState) = 0.0
ref_rho_c(rs::CondensateReferenceState) = rs.rho_cbar

"""Saturation-ratio reference profile `(nlevels, 3)` (`q_v/q_sat`); `0.0` if absent."""
ref_sat(rs::DryReferenceState) = 0.0
ref_sat(rs::MoistReferenceState) = rs.satbar
ref_sat(rs::CondensateReferenceState) = rs.satbar

ref_rho_v(rs::PressureReferenceState) = rs.rho_vbar
ref_rho_c(rs::PressureReferenceState) = rs.rho_cbar

"""Total-pressure reference profile `(nlevels, 3)` [Pa] (PressureReferenceState only)."""
ref_pressure(rs::PressureReferenceState) = rs.pbar

"""Total-density reference profile `(nlevels, 3)` [kg/m^3] (PressureReferenceState only)."""
ref_rho_t(rs::PressureReferenceState) = rs.rho_tbar

"""Total-energy-density reference profile `(nlevels, 3)` [J/m^3] (PressureReferenceState only)."""
ref_total_energy(rs::PressureReferenceState) = rs.E_tbar

"""Supersaturation-density reference profile `(nlevels, 3)` [kg/m^3] (PressureReferenceState only)."""
ref_qss(rs::PressureReferenceState) = rs.Q_ssbar

"""
    reference_temperature(rs) -> Vector{Float64}

Temperature profile [K] recovered from the stored physical state. Used for diagnostics
and (in Daisho) hydrometeor classification.
"""
function reference_temperature(rs::AbstractReferenceState)
    s = ref_entropy(rs)[:, 1]
    rho_d = ref_rho_d(rs)[:, 1]
    rv = ref_rho_v(rs)
    q_v = rv === 0.0 ? zero(rho_d) : rv[:, 1] ./ rho_d
    return temperature.(s, rho_d, q_v)
end

reference_temperature(rs::PressureReferenceState) = rs.Tbar[:, 1]

# ── Column helpers ─────────────────────────────────────────────────────────────

"""
    reference_column(grid::AbstractGrid, grid_params)

Build a vertical basis column with natural (R0) boundary conditions for reference-state
derivative calculations. Reference profiles can have nonzero boundary gradients, so the
model variables' wall BCs must not be imposed on them.
"""
function reference_column(grid::AbstractGrid, grid_params)
    return natural_column(grid.kbasis.data[1], grid_params)
end

function natural_column(column::Chebyshev1D, grid_params)
    cp = ChebyshevParameters(
        zmin = grid_params.kMin,
        zmax = grid_params.kMax,
        zDim = grid_params.kDim,
        bDim = grid_params.b_kDim,
        BCB = Chebyshev.R0,
        BCT = Chebyshev.R0)
    return Chebyshev1D(cp)
end

function natural_column(column::Spline1D, grid_params)
    sp = SplineParameters(
        xmin = grid_params.kMin,
        xmax = grid_params.kMax,
        num_cells = grid_params.num_cells_k,
        mubar = grid_params.mubar,
        quadrature = grid_params.quadrature,
        BCL = CubicBSpline.R0,
        BCR = CubicBSpline.R0)
    return Spline1D(sp)
end

function natural_column(column, grid_params)
    return deepcopy(column)
end

"""
    transform_reference_state!(column, ref::Array{Float64})

Compute vertical derivatives of a reference profile in place. Fits `ref[:,1]` to the
spectral basis, then overwrites `ref[:,1]` with the filtered values, `ref[:,2]` with
the first vertical derivative, and `ref[:,3]` with the second.
"""
function transform_reference_state!(column, ref::Array{Float64})
    column.uMish[:] .= ref[:, 1]
    Btransform!(column)
    Atransform!(column)
    ref[:, 1] .= Itransform!(column)
    ref[:, 2] .= Ixtransform(column)
    ref[:, 3] .= Ixxtransform(column)
    return ref
end

# Fit a value vector to the column and return a fresh (nlevels, 3) profile of
# [filtered value, d/dz, d2/dz2].
function _profile(column, values::AbstractVector{Float64})
    prof = zeros(Float64, length(values), 3)
    prof[:, 1] .= values
    transform_reference_state!(column, prof)
    return prof
end

# ── Builders ───────────────────────────────────────────────────────────────────

# Read a sounding file (surface line `p_sfc theta q_v`, then `z theta q_v` per level;
# q_v in g/kg) into (sfc_pressure, alt, theta_in, q_v_in_kgkg).
function _read_sounding(ref_state_file::AbstractString)
    alt = Float64[]
    theta_in = Float64[]
    q_v_in = Float64[]
    sfc_pressure = 0.0
    open(ref_state_file, "r") do ref
        surface = readline(ref)
        sfc_pressure = parse(Float64, split(surface)[1])
        push!(alt, 0.0)
        push!(theta_in, parse(Float64, split(surface)[2]))
        push!(q_v_in, parse(Float64, split(surface)[3]))
        while true
            level = readline(ref)
            isempty(level) && break
            push!(alt, parse(Float64, split(level)[1]))
            push!(theta_in, parse(Float64, split(level)[2]))
            push!(q_v_in, parse(Float64, split(level)[3]))
        end
    end
    return sfc_pressure, alt, theta_in, q_v_in
end

# Piecewise-linear interpolation of `vals` (defined at `alt`) to model levels `z`.
function _interp_to_levels(z, alt, vals)
    out = zeros(Float64, length(z))
    out[1] = vals[1]
    for i in 2:length(z)
        found = false
        for j in 2:length(alt)
            if (alt[j-1] < z[i]) && (alt[j] > z[i])
                out[i] = vals[j-1] + (z[i] - alt[j-1]) * (vals[j] - vals[j-1]) / (alt[j] - alt[j-1])
                found = true
            elseif alt[j] == z[i]
                out[i] = vals[j]
                found = true
            end
        end
        found || throw(DomainError(i, "Can't find an interpolating level for reference state"))
    end
    return out
end

"""
    calculate_reference_state(ref_state_file, z, column; moisture=true)

Build a hydrostatic reference state from a sounding file (`theta`, `q_v`), interpolating
to model levels `z` and re-integrating spectrally on `column` to hydrostatic balance,
then refining the balance with a short Newton-style iteration. Returns a
[`MoistReferenceState`](@ref) (`moisture=true`) or a [`DryReferenceState`](@ref).

This integrates in physical density (log-density) and differentiates `q_v` directly,
so the result is independent of any prognostic-variable (`mu`) transform. With the
linear `mu = q_v·1e5` convention this reproduces the historical `xi`/`mu` reference to
round-off.
"""
function calculate_reference_state(ref_state_file::AbstractString, z::Array{Float64}, column;
                                   moisture::Bool=true)

    sfc_pressure, alt, theta_in, q_v_in = _read_sounding(ref_state_file)
    nz = length(z)

    # Vertical finite-difference derivatives of the sounding
    qvdz = zeros(Float64, length(alt))
    thetadz = zeros(Float64, length(alt))
    qvdz[1] = (q_v_in[2] - q_v_in[1]) / alt[2]
    thetadz[1] = (theta_in[2] - theta_in[1]) / alt[2]
    for i in 2:(length(alt)-1)
        qvdz[i] = (q_v_in[i+1] - q_v_in[i-1]) / (alt[i+1] - alt[i-1])
        thetadz[i] = (theta_in[i+1] - theta_in[i-1]) / (alt[i+1] - alt[i-1])
    end
    qvdz[end] = (q_v_in[end] - q_v_in[end-1]) / (alt[end] - alt[end-1])
    thetadz[end] = (theta_in[end] - theta_in[end-1]) / (alt[end] - alt[end-1])

    # Interpolate the derivative profiles to model levels (value at level 1 = sfc deriv)
    thetadz_z = _interp_to_levels(z, alt, thetadz)
    qvdz_z = _interp_to_levels(z, alt, qvdz)
    thetadz_z[1] = thetadz[1]
    qvdz_z[1] = qvdz[1]

    # Spectrally integrate the derivative profiles to get smoothed theta and q_v
    column.uMish[:] .= thetadz_z[:]
    Btransform!(column); Atransform!(column)
    theta_new = IInttransform(column, theta_in[1])

    column.uMish[:] .= (qvdz_z .* 1.0e-3)[:]
    Btransform!(column); Atransform!(column)
    q_v_new = IInttransform(column, q_v_in[1] * 1.0e-3)

    # Smooth q_v and obtain its vertical derivative directly (no mu transform). Floor
    # at zero: spectral representation of a (near-)zero vapor field can dip slightly
    # negative, which is unphysical (the mu-transform path floored at zero implicitly
    # via inv_mu_transform).
    column.uMish[:] .= q_v_new[:]
    Btransform!(column); Atransform!(column)
    q_v_new = max.(0.0, Itransform!(column))
    q_v_new_z = Ixtransform(column)

    # Hydrostatic pressure and density from theta and q_v via the Exner function
    theta_rho = @. theta_new * (1.0 + (q_v_new / Eps)) / (1.0 + q_v_new)
    dexnerdz = -gravity ./ (Cpd .* theta_rho)
    column.uMish[:] .= dexnerdz
    Btransform!(column); Atransform!(column)
    sfc_exner = (sfc_pressure / 1000.0)^(Rd / Cpd)
    exner = IInttransform(column, sfc_exner)
    p_new = @. (exner^(Cpd / Rd)) * 1000.0
    rho_t_new = @. ((p_new * 100.0 / (Rd * theta_rho)) * (1000.0 / p_new)^(Rd / Cpd))
    rho_d_new = rho_t_new ./ (1.0 .+ q_v_new)
    xi_new = log_dry_density.(rho_d_new)
    sfc_xi = xi_new[1]

    # Moist entropy from the hydrostatic temperature
    Tk_new = @. (p_new - vapor_pressure(p_new, q_v_new)) * 100.0 / (rho_d_new * Rd)
    s_new = entropy.(Tk_new, rho_d_new, q_v_new)
    column.uMish[:] .= s_new
    Btransform!(column); Atransform!(column)
    s_new = copy(Itransform!(column))   # copy: Itransform! returns a reused buffer
    s_new_z = Ixtransform(column)
    Tk_new = temperature.(s_new, rho_d_new, q_v_new)

    # Refine the hydrostatic balance: -g*rho_t = P_s*s_z + P_xi*xi_z + P_qv*q_v_z
    for _ in 1:10
        Ps = P_s.(Tk_new, rho_d_new, q_v_new)
        Pxi = P_xi.(Tk_new, rho_d_new, q_v_new)
        Pqv = P_qv.(Tk_new, rho_d_new, q_v_new)
        xi_new_z = ((-gravity .* rho_t_new) .- (Ps .* s_new_z) .- (Pqv .* q_v_new_z)) ./ Pxi
        column.uMish[:] .= xi_new_z[:]
        Btransform!(column); Atransform!(column)
        xi_new = IInttransform(column, sfc_xi)
        rho_d_new = dry_density.(xi_new)
        rho_t_new = rho_d_new .* (1.0 .+ q_v_new)
        Tk_new = temperature.(s_new, rho_d_new, q_v_new)
    end

    # Assemble physical profiles with spectrally consistent derivatives
    sbar = _profile(column, s_new)
    sigmabar = _profile(column, rho_d_new .* s_new)   # entropy density on the reference basis
    rho_dbar = _profile(column, rho_d_new)
    sound = _mean_sound_speed_sq(Tk_new, rho_d_new, q_v_new)

    if !moisture
        return DryReferenceState(sbar, sigmabar, rho_dbar, sound)
    end

    rho_vbar = _profile(column, rho_d_new .* q_v_new)
    # Saturation ratio q_v/q_sat from the post-refinement consistent pressure (rho_d
    # changed during the hydrostatic iteration, so recompute p rather than reuse the
    # pre-loop Exner pressure).
    p_final = pressure.(s_new, rho_d_new, q_v_new)
    q_sat = q_sat_liquid.(Tk_new, p_final)
    satbar = _profile(column, q_v_new ./ q_sat)
    return MoistReferenceState(sbar, sigmabar, rho_dbar, rho_vbar, satbar, sound)
end

"""
    interpolate_reference_state(ref_state_file, z, column; moisture=true)

Build a reference state from a sounding by simple log-pressure hydrostatic integration
(no spectral re-integration of the raw profiles); vertical derivatives are computed
afterwards spectrally. Lighter-weight alternative to [`calculate_reference_state`](@ref).
"""
function interpolate_reference_state(ref_state_file::AbstractString, z::Array{Float64}, column;
                                     moisture::Bool=true)

    sfc_pressure, alt, theta_in, q_v_in = _read_sounding(ref_state_file)

    theta = _interp_to_levels(z, alt, theta_in)
    q_v = _interp_to_levels(z, alt, q_v_in) .* 1.0e-3
    nlevels = length(z)

    Tk = zeros(Float64, nlevels)
    p = zeros(Float64, nlevels)
    rho_d = zeros(Float64, nlevels)

    p[1] = sfc_pressure
    e = vapor_pressure(p[1], q_v[1])
    Tk[1] = theta[1] / (p_0 / p[1])^(Rd / Cpd)
    rho_d[1] = 100.0 * (p[1] - e) / (Tk[1] * Rd)
    rho_t = rho_d[1] * (1.0 + q_v[1])
    dlnpdz = -gravity * rho_t / (p[1] * 100.0)
    for i in 2:nlevels
        p[i] = exp(log(p[i-1]) + (dlnpdz * (z[i] - z[i-1])))
        Tk[i] = theta[i] / (p_0 / p[i])^(Rd / Cpd)
        e = vapor_pressure(p[i], q_v[i])
        rho_d[i] = 100.0 * (p[i] - e) / (Tk[i] * Rd)
        rho_t = rho_d[i] * (1.0 + q_v[i])
        dlnpdz = -gravity * rho_t / (p[i] * 100.0)
    end

    s = entropy.(Tk, rho_d, q_v)
    sbar = _profile(column, s)
    sigmabar = _profile(column, rho_d .* s)
    rho_dbar = _profile(column, rho_d)
    sound = _mean_sound_speed_sq(Tk, rho_d, q_v)

    if !moisture
        return DryReferenceState(sbar, sigmabar, rho_dbar, sound)
    end

    rho_vbar = _profile(column, rho_d .* q_v)
    q_sat = q_sat_liquid.(Tk, p)
    satbar = _profile(column, q_v ./ q_sat)
    return MoistReferenceState(sbar, sigmabar, rho_dbar, rho_vbar, satbar, sound)
end

"""
    exact_reference_state(ref_state_file, z, column) -> CondensateReferenceState

Read a pre-balanced reference state from a physical-density file with one line per model
level: `z s rho_d rho_v rho_c` (moist entropy, dry-air, vapor, and condensate partial
densities). Vertical derivatives and the saturation-ratio profile are computed on
`column`. Used for highly idealized, already-balanced base states (e.g. the saturated
Bryan & Fritsch 2002 benchmark, where a nonzero `rho_c` makes the base neutrally
buoyant). The `z` values must match the model levels (compared as written by
`string`).
"""
function exact_reference_state(ref_state_file::AbstractString, z::Array{Float64}, column)
    n = length(z)
    s = zeros(Float64, n); rho_d = zeros(Float64, n)
    rho_v = zeros(Float64, n); rho_c = zeros(Float64, n)
    open(ref_state_file, "r") do f
        for i in 1:n
            parts = split(readline(f))
            parts[1] == string(z[i]) ||
                throw(DomainError(i, "Model level does not match reference level"))
            s[i] = parse(Float64, parts[2])
            rho_d[i] = parse(Float64, parts[3])
            rho_v[i] = parse(Float64, parts[4])
            rho_c[i] = parse(Float64, parts[5])
        end
    end

    sbar = _profile(column, s)
    sigmabar = _profile(column, rho_d .* s)
    rho_dbar = _profile(column, rho_d)
    rho_vbar = _profile(column, rho_v)
    rho_cbar = _profile(column, rho_c)

    q_v = rho_v ./ rho_d
    Tk = temperature.(s, rho_d, q_v)
    p = pressure.(s, rho_d, q_v)
    q_sat = q_sat_liquid.(Tk, p)
    satbar = _profile(column, q_v ./ q_sat)
    sound = _mean_sound_speed_sq(Tk, rho_d, q_v)

    return CondensateReferenceState(sbar, sigmabar, rho_dbar, rho_vbar, rho_cbar, satbar, sound)
end

# Domain-mean speed of sound squared, c^2 = (dp/drho)|_s ~ P_xi / rho_t, averaged.
function _mean_sound_speed_sq(Tk, rho_d, q_v)
    c2 = P_xi.(Tk, rho_d, q_v) ./ (rho_d .* (1.0 .+ q_v))
    return sum(c2) / length(c2)
end

"""
    _hydrostatic_pressure_profile(column, rho_tbar, p_anchor; tol=0.01) -> Matrix

Pressure profile `(nlevels, 3)` in EXACT discrete hydrostatic balance with the fitted
total density `rho_tbar` (itself an `(nlevels, 3)` profile), anchored so the value at the
first mish point is `p_anchor`.

Fitting `p` directly and reading back its spline derivative — which is what `_profile`
does — CANNOT resolve `dp/dz` in the upper atmosphere. The fit is accurate to ~0.03 %,
but 0.03 % of 1e5 Pa is ~30 Pa, and across a 500 m cell that is ~0.06 Pa/m, which is
10-17 % of `g·ρ_t` where `p` is small. Measured on the 50-cell / 25 km tropical-cyclone
grid, the stored `-(dp̄/dz + g·ρ̄_t)/ρ̄_t` reached **+1.67 m/s²** at 22.6 km — 300x the
balanced vortex's own hydrostatic residual — switching on at the tropopause and growing
monotonically to the lid. It is a cancellation/scale problem, not the spline operator
(a centred finite difference of the same `p̄` agrees with the spline derivative to <1 %)
and not the `l_q` smoothing (identical to 4 digits at l_q = 2.0, 1.0, 0.5, 0.1 and 0.0).

Two steps, in order:

1. **Integrate, do not re-fit.** Fit `-g·ρ̄_t` (accurate to 0.03 %) and take its
   antiderivative with `IInttransform`. The value slot is that antiderivative — the
   spline whose derivative IS the fit of `-g·ρ̄_t` by construction. The previous code
   built exactly this pressure and then threw the spline away, re-fitting its VALUES.
2. **Snap the derivative slots** to `-g·ρ̄_t` and `-g·dρ̄_t/dz`, so the discrete balance
   is exact to machine precision rather than to fit accuracy. Step 1 is what makes this
   a ~0.03 % adjustment instead of a 17 % one, i.e. what keeps the value slot and the
   derivative slots mutually consistent.

The reconstruction is then compared against the input pressure and rejected above
`tol` (relative). That is the coarse-sounding gate: a sounding too coarse to support a
discrete hydrostatic state on the requested grid must fail loudly, not silently produce
an unbalanced reference.
"""
function _hydrostatic_pressure_profile(column, rho_tbar::Array{Float64},
                                       p_anchor::AbstractVector{Float64}; tol::Float64=0.01)
    n = size(rho_tbar, 1)
    prof = zeros(Float64, n, 3)

    # (1) antiderivative of the FITTED -g*rho_t; anchor afterwards, since IInttransform's
    # C0 pins the value at `xmin`, which is not a mish point.
    column.uMish[:] .= -gravity .* view(rho_tbar, :, 1)
    Btransform!(column)
    Atransform!(column)
    p_int = IInttransform(column, 0.0)
    prof[:, 1] .= p_int .+ (p_anchor[1] - p_int[1])

    # (2) exact discrete balance
    prof[:, 2] .= -gravity .* view(rho_tbar, :, 1)
    prof[:, 3] .= -gravity .* view(rho_tbar, :, 2)

    # The coarse-sounding gate
    err, k = findmax(abs.(prof[:, 1] .- p_anchor) ./ p_anchor)
    err <= tol || throw(DomainError(err,
        "the input sounding does not support a discrete hydrostatic state on this " *
        "grid: re-integrating dp/dz = -g*rho_t from the fitted density reproduces the " *
        "input pressure to only $(round(100*err, sigdigits=3)) % (tolerance " *
        "$(round(100*tol, sigdigits=3)) %), worst at level $k, " *
        "p_in = $(p_anchor[k]) Pa vs p_hydrostatic = $(prof[k, 1]) Pa. Refine the " *
        "sounding's vertical resolution or coarsen the model grid."))
    return prof
end

# Assemble a PressureReferenceState from pointwise (p [Pa], rho_d, rho_v, rho_c) on
# levels z: EOS temperature, BF02 total-energy density at rest, supersaturation
# density, and mean gamma*p/rho_t sound speed; profiles fitted on `column`.
#
# `hydrostatic = true` (opt-in; off by default and bitwise inert when off) builds the
# pressure profile through `_hydrostatic_pressure_profile` instead of a plain fit, so
# the stored dp̄/dz satisfies discrete hydrostatic balance EXACTLY. The equation sets
# that consume a PressureReferenceState carry `dp̄/dz = -g·ρ̄_t` as an unstated
# assumption — every reference-derivative term in the moist_compressible tendencies
# multiplies `w`, and the `w` forcing is perturbation-only — so where the stored
# derivative violates it, the model silently omits a forcing of that size.
function _pressure_reference(z, column, p_Pa, rho_d, rho_v, rho_c; hydrostatic::Bool=false)
    rho_t = rho_d .+ rho_v .+ rho_c
    Tk = p_Pa ./ ((rho_d .* Rd) .+ (rho_v .* Rv))
    q_v = rho_v ./ rho_d
    q_l = rho_c ./ rho_d
    E_t = (rho_d .* internal_energy_bf02.(Tk, q_v, q_l)) .+ (rho_t .* gravity .* z)
    Q_ss = rho_v .- rho_v_sat.(Tk, p_Pa ./ 100.0)
    C_vt = Cvd .+ (q_v .* Cvv) .+ (q_l .* Cl)
    R_m = Rd .+ (q_v .* Rv)
    gamma = (C_vt .+ R_m) ./ C_vt
    sound = sum(gamma .* p_Pa ./ rho_t) / length(p_Pa)
    rho_tbar = _profile(column, rho_t)
    pbar = hydrostatic ? _hydrostatic_pressure_profile(column, rho_tbar, p_Pa) :
                         _profile(column, p_Pa)
    return PressureReferenceState(
        pbar, _profile(column, rho_d), _profile(column, rho_v),
        _profile(column, rho_c), rho_tbar, _profile(column, Tk),
        _profile(column, E_t), _profile(column, Q_ss), sound)
end

"""
    exact_pressure_reference_state(ref_state_file, z, column) -> PressureReferenceState

Read a pre-balanced pressure-based reference state from a file with one line per model
level: `z p rho_d rho_v rho_c` (total pressure in Pa, then dry-air, vapor, and
condensate partial densities). Temperature comes directly from the equation of state
(no entropy inversion); the total-energy density and supersaturation density are
derived pointwise before fitting, so a saturated input column has `Q_ssbar = 0`
identically. The `z` values must match the model levels (compared as written by
`string`).

`hydrostatic = true` rebuilds the pressure profile from the fitted density so the
stored `dp̄/dz` is in EXACT discrete hydrostatic balance (see
`_hydrostatic_pressure_profile`); the file's pressure is then used only as the
integration anchor and as the accuracy check. This is what lets the values-only file
round-trip carry a balanced reference without a format change. Off by default and
bitwise inert when off.
"""
function exact_pressure_reference_state(ref_state_file::AbstractString, z::Array{Float64}, column;
                                        hydrostatic::Bool=false)
    n = length(z)
    p_Pa = zeros(Float64, n); rho_d = zeros(Float64, n)
    rho_v = zeros(Float64, n); rho_c = zeros(Float64, n)
    open(ref_state_file, "r") do f
        for i in 1:n
            parts = split(readline(f))
            parts[1] == string(z[i]) ||
                throw(DomainError(i, "Model level does not match reference level"))
            p_Pa[i] = parse(Float64, parts[2])
            rho_d[i] = parse(Float64, parts[3])
            rho_v[i] = parse(Float64, parts[4])
            rho_c[i] = parse(Float64, parts[5])
        end
    end
    return _pressure_reference(z, column, p_Pa, rho_d, rho_v, rho_c; hydrostatic)
end

"""
    calculate_pressure_reference_state(ref_state_file, z, column) -> PressureReferenceState

Build a pressure-based hydrostatic reference from a sounding file (`theta`, `q_v`;
same format as [`calculate_reference_state`](@ref)). The balance is the direct
dp/dz = −ρ_t g, integrated spectrally on `column` with a short fixed-point sweep —
no entropy/log-density (P_s, P_xi, P_qv) Newton refinement is required. Condensate-free
(`rho_cbar = 0`).

`hydrostatic = true` keeps the antiderivative spline instead of re-fitting its values,
and snaps the derivative slots, so the stored `dp̄/dz` is in EXACT discrete hydrostatic
balance (see `_hydrostatic_pressure_profile`). Off by default and bitwise inert
when off.
"""
function calculate_pressure_reference_state(ref_state_file::AbstractString, z::Array{Float64}, column;
                                            hydrostatic::Bool=false)

    sfc_pressure, alt, theta_in, q_v_in = _read_sounding(ref_state_file)

    theta = _interp_to_levels(z, alt, theta_in)
    q_v = _interp_to_levels(z, alt, q_v_in) .* 1.0e-3
    nlevels = length(z)

    # Initial guess: level-by-level log-pressure march (as interpolate_reference_state)
    Tk = zeros(Float64, nlevels)
    p = zeros(Float64, nlevels)      # hPa
    rho_d = zeros(Float64, nlevels)
    p[1] = sfc_pressure
    e = vapor_pressure(p[1], q_v[1])
    Tk[1] = theta[1] / (p_0 / p[1])^(Rd / Cpd)
    rho_d[1] = 100.0 * (p[1] - e) / (Tk[1] * Rd)
    dlnpdz = -gravity * rho_d[1] * (1.0 + q_v[1]) / (p[1] * 100.0)
    for i in 2:nlevels
        p[i] = exp(log(p[i-1]) + (dlnpdz * (z[i] - z[i-1])))
        Tk[i] = theta[i] / (p_0 / p[i])^(Rd / Cpd)
        e = vapor_pressure(p[i], q_v[i])
        rho_d[i] = 100.0 * (p[i] - e) / (Tk[i] * Rd)
        dlnpdz = -gravity * rho_d[i] * (1.0 + q_v[i]) / (p[i] * 100.0)
    end

    # Fixed-point refinement: spectrally integrate dp/dz = -g*rho_t, then update
    # T (theta, Exner), rho_d (EOS) from the new pressure.
    #
    # The sweep OSCILLATES before it settles — it is a damped alternation, not a
    # monotone approach — so a fixed count leaves the returned (p, rho_d) pair
    # inconsistent by wherever the swing happened to be. On the 50-cell / 25 km
    # tropical-cyclone grid with the Dunion MT sounding, the lid pressure runs
    #
    #   it     1      2      3      4      5      6      7      8     ...    50
    #   p   1912   4010   1512   3550   2273   2897   2642   2724     ...  2707 Pa
    #
    # and the historical `1:5` stops on 2273 Pa — 16 % BELOW the converged 2707 Pa,
    # with the pair mutually inconsistent to 27 %. That truncation, not the sounding's
    # vertical resolution and not the spline operator, is why the stored dp̄/dz departed
    # from -g·ρ̄_t by up to 17 % above the tropopause.
    #
    # `hydrostatic = true` iterates to convergence instead. The legacy `1:5` is kept as
    # the default so every existing baseline stays bitwise unchanged.
    p_Pa = p .* 100.0
    maxit = hydrostatic ? 200 : 5
    converged = !hydrostatic
    for it in 1:maxit
        rho_t = rho_d .* (1.0 .+ q_v)
        column.uMish[:] .= (-gravity .* rho_t)[:]
        Btransform!(column); Atransform!(column)
        p_next = IInttransform(column, sfc_pressure * 100.0)
        delta = maximum(abs.(p_next .- p_Pa) ./ p_Pa)
        p_Pa = p_next
        p = p_Pa ./ 100.0
        Tk = theta ./ (p_0 ./ p).^(Rd / Cpd)
        e_v = vapor_pressure.(p, q_v)
        rho_d = 100.0 .* (p .- e_v) ./ (Tk .* Rd)
        if hydrostatic && it > 1 && delta < 1.0e-12
            converged = true
            break
        end
    end
    converged || throw(DomainError(maxit,
        "the hydrostatic fixed point did not converge in $maxit sweeps for " *
        "'$ref_state_file' on this grid. The sounding's (theta, q_v) profile and the " *
        "hydrostatic integral are not mutually consistent to 1e-12."))

    rho_v = rho_d .* q_v
    rho_c = zeros(Float64, nlevels)
    return _pressure_reference(z, column, p_Pa, rho_d, rho_v, rho_c; hydrostatic)
end
