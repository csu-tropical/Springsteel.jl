"""
    Springsteel.Thermodynamics

Basis-agnostic atmospheric thermodynamics following Emanuel (1994), shared by the
codes that build on Springsteel grids (Scythe, Daisho, …). This submodule holds the
*physical* equation of state and standard diagnostics expressed in physical state
variables `(s, rho_d, q_v)` (moist entropy, dry-air density, vapor mixing ratio) and
liquid mixing ratio `q_l` where relevant. Prognostic-variable transforms (the `xi`/`mu`
control variables and their pressure derivatives) are model-specific and stay in the
client code.

| Constant | Value | Units | Description |
|:---------|:------|:------|:------------|
| `Rd`     | 287.04 | J/(kg·K) | Gas constant for dry air |
| `Rv`     | 461.50 | J/(kg·K) | Gas constant for water vapor |
| `Eps`    | Rd/Rv  | – | Ratio of gas constants |
| `Cvd`    | 716.96 | J/(kg·K) | Specific heat of dry air at constant volume |
| `Cvv`    | 1410.0 | J/(kg·K) | Specific heat of water vapor at constant volume |
| `Cpd`    | Cvd+Rd | J/(kg·K) | Specific heat of dry air at constant pressure |
| `Cpv`    | Cvv+Rv | J/(kg·K) | Specific heat of water vapor at constant pressure |
| `Cl`     | 4186.0 | J/(kg·K) | Specific heat of liquid water |
| `Ci`     | 2106.0 | J/(kg·K) | Specific heat of ice |
| `gravity`| 9.81   | m/s² | Gravitational acceleration |
| `L_v0`   | 2.501e6 | J/kg | Latent heat of vaporization at T₀ |
| `rho_l`  | 1000.0 | kg/m³ | Density of liquid water |
| `rho_i`  | 917.0  | kg/m³ | Density of ice |
| `T_0`    | 273.16 | K | Reference temperature (triple point of water) |
| `p_0`    | 1000.0 | hPa | Reference pressure |
| `rho_d0` | 100·p₀/(T₀·Rd) | kg/m³ | Reference dry air density |
| `rho_v0` | 100·eₛ(T₀)/(T₀·Rv) | kg/m³ | Reference vapor density at T₀ |

# References
- Emanuel, K. A. (1994). *Atmospheric Convection*. Oxford University Press.
"""
module Thermodynamics

export Rd, Rv, Eps, Cvd, Cvv, Cpd, Cpv, Cl, Ci, gravity, L_v0, rho_l, rho_i,
    T_0, p_0, q0, rho_d0, rho_v0
export sat_pressure_liquid, sat_pressure_ice, sat_pressure_liquid_buck,
    sat_pressure_liquid_buck_dT, sat_pressure_ice_buck, q_sat_liquid, q_sat_ice,
    L_v, dewpoint, entropy, vapor_entropy, temperature, pressure, vapor_pressure,
    mixing_ratio, dry_density, log_dry_density, P_s, P_xi, P_qv, P_rhod, P_rhov,
    potential_temperature, reversible_theta_e, theta_rho,
    rho_v_sat, internal_energy_bf02

# Constants from Emanuel (1994)
const Rd = 287.04
const Rv = 461.50
const Eps = Rd / Rv
const Cvd = 716.96
const Cvv = 1410.0
const Cpd = Cvd + Rd
const Cpv = Cvv + Rv
const Cl = 4186.0
const Ci = 2106.0 # Ice heat capacity
const gravity = 9.81
const L_v0 = 2.501e6
const rho_l = 1000.0 # Density of liquid water in kg/m^3
const rho_i = 917.0 # Density of ice in kg/m^3

# Entropy function constants
const T_0 = 273.16
const p_0 = 1000.0
const q0 = 1.0e-5

"""
    sat_pressure_liquid(Tk)

Saturation vapor pressure over liquid water [hPa] from the Bolton (1980) formula.

# References
- Bolton, D. (1980). *Mon. Wea. Rev.*, 108, 1046–1053.
"""
function sat_pressure_liquid(Tk::Float64)

    Tc = Tk - 273.15
    return 6.112 * exp(17.67 * Tc / (Tc + 243.5))
end

"""
    sat_pressure_ice(Tk)

Saturation vapor pressure over ice [hPa].
"""
function sat_pressure_ice(Tk::Float64)

    Tc = Tk - 273.15
    return 6.112 * exp(21.8745584 * Tc / (Tc + 265.49))
end

const rho_d0 = 100.0 * p_0 / (T_0 * Rd)
const rho_v0 = 100.0 * sat_pressure_liquid(T_0) / (T_0 * Rv)

"""
    L_v(Tk)

Latent heat of vaporization [J/kg] as a linear function of temperature.

# References
- Emanuel, K. A. (1994). *Atmospheric Convection*. Oxford University Press.
"""
function L_v(Tk::Float64)

    return L_v0 + ((Cpv - Cl) * (Tk - T_0))
end

"""
    entropy(Tk, rho_d, q_v)

Moist entropy per unit mass of dry air [J/(kg·K)] from temperature, dry-air density,
and water-vapor mixing ratio, using the Emanuel (1994) formulation.
"""
function entropy(Tk::Float64, rho_d::Float64, q_v::Float64)

    qfactor = 0.0
    if (q_v != 0.0)
        qfactor = q_v * (Rv * log(q_v * rho_d / rho_v0) - (L_v(T_0)/T_0))
    end

    Cfactor = Cvd + (q_v * Cvv)
    s = (Cfactor * log(Tk/T_0)) - (Rd * log(rho_d/rho_d0)) - qfactor
    return s
end

"""
    vapor_entropy(Tk, rho_d, q_v)

Water-vapor contribution to specific entropy [J/(kg·K)]; zero for non-positive `q_v`.
"""
function vapor_entropy(Tk::Float64, rho_d::Float64, q_v::Float64)

    if q_v > 0.0
        return (Cvv * log(Tk/T_0)) - (Rv * log(q_v * rho_d / rho_v0)) + (L_v(T_0)/T_0)
    else
        return 0.0
    end
end

"""
    temperature(s, rho_d, q_v)

Temperature [K] recovered from moist entropy, dry-air density, and vapor mixing ratio
by inverting [`entropy`](@ref).
"""
function temperature(s::Float64, rho_d::Float64, q_v::Float64)

    Cfactor = Cvd + (q_v * Cvv)
    qfactor = 1.0
    if (q_v != 0.0)
        qfactor = (rho_d * q_v / rho_v0)^((q_v * Rv) / Cfactor)
    end

    rhofactor = (rho_d / rho_d0)^(Rd / Cfactor)
    Tfactor = exp((s - (q_v * L_v(T_0)/T_0)) / Cfactor)

    T = T_0 * Tfactor * rhofactor * qfactor
    return T
end

"""
    pressure(s, rho_d, q_v)

Total pressure (dry air + vapor) [hPa] from moist entropy, dry-air density, and vapor
mixing ratio.
"""
function pressure(s::Float64, rho_d::Float64, q_v::Float64)

    Tk = temperature(s, rho_d, q_v)
    pd = 0.01 * Rd * Tk * rho_d
    e = 0.01 * Rv * Tk * rho_d * q_v
    return pd + e
end

"""
    vapor_pressure(p, q_v)

Partial pressure of water vapor [hPa] from total pressure [hPa] and mixing ratio.
"""
function vapor_pressure(p::Float64, q_v::Float64)

    e = (p * q_v)/(Eps + q_v)
end

"""
    mixing_ratio(p, e)

Water-vapor mixing ratio [kg/kg] from total pressure and vapor pressure [hPa].
"""
function mixing_ratio(p::Float64, e::Float64)

    q_v = (Eps * e)/(p-e)
end

"""
    dewpoint(p, q_v)

Dewpoint temperature [K] from total pressure [hPa] and vapor mixing ratio, by
inverting the Bolton (1980) saturation vapor pressure.
"""
function dewpoint(p::Float64, q_v::Float64)

    e = vapor_pressure(p, q_v)
    Tc = 243.5 * log(e/6.112) / (17.67 - log(e/6.112))
    return Tc + 273.15
end

"""
    sat_pressure_liquid_buck(Tk, phPa)

Saturation vapor pressure over liquid water [hPa] from Buck (1981), including the
dry-air pressure enhancement factor.

# References
- Buck, A. L. (1981). *J. Appl. Meteor.*, 20, 1527–1532.
"""
function sat_pressure_liquid_buck(Tk::Float64, phPa::Float64)

    Tc = Tk - 273.15
    A = 7.2e-4
    B = 3.20e-6
    C = 5.9e-10
    fw4 = 1.0 + A + (phPa * (B + (C * Tc^2)))

    a = 6.1121
    b = 18.729
    c = 257.87
    d = 227.3
    ew4 = a * exp( (b - (Tc / d)) * Tc / (Tc + c) )

    return fw4 * ew4
end

"""
    sat_pressure_liquid_buck_dT(Tk, phPa)

Derivative of the Buck (1981) saturation vapor pressure over liquid with respect to
temperature at constant pressure [hPa/K].
"""
function sat_pressure_liquid_buck_dT(Tk::Float64, phPa::Float64)

    Tc = Tk - 273.15

    A = 7.2e-4
    B = 3.20e-6
    C = 5.9e-10
    fw4 = 1.0 + A + (phPa * (B + (C * Tc^2)))
    d_fw4 = 2.0 * phPa * C * Tc

    a = 6.1121
    b = 18.729
    c = 257.87
    d = 227.3
    ew4 = a * exp( (b - (Tc / d)) * Tc / (Tc + c) )
    T1 = (d * b - (2.0 * Tc)) * (d * (Tc + c)) - d* ((d * b * Tc) - Tc^2)
    T2 =  (d * (Tc + c))^2
    d_ew4 = ew4 * T1 / T2

    return ew4 * d_fw4 + fw4 * d_ew4
end

"""
    sat_pressure_ice_buck(Tk, phPa)

Saturation vapor pressure over ice [hPa] from Buck (1981), including the dry-air
pressure enhancement factor.
"""
function sat_pressure_ice_buck(Tk::Float64, phPa::Float64)

    Tc = Tk - 273.15
    A = 2.2e-4
    B = 3.83e-6
    C = 6.4e-10
    fi4 = 1.0 + A + (phPa * (B + (C * Tc^2)))

    a = 6.1115
    b = 23.036
    c = 279.82
    d = 333.7
    ei3 = a * exp( (b - (Tc / d)) * Tc / (Tc + c) )

    return fi4 * ei3
end

"""
    q_sat_liquid(Tk, phPa)

Saturation mixing ratio over liquid water [kg/kg] from the Buck (1981) formula.
"""
function q_sat_liquid(Tk::Float64, phPa::Float64)

    ew = sat_pressure_liquid_buck(Tk,phPa)
    q_sat = Eps * ew / (phPa - ew)
    return q_sat
end

"""
    q_sat_ice(Tk, phPa)

Saturation mixing ratio over ice [kg/kg] from the Buck (1981) formula.
"""
function q_sat_ice(Tk::Float64, phPa::Float64)

    ei = sat_pressure_ice_buck(Tk,phPa)
    q_sat = Eps * ei / (phPa - ei)
    return q_sat
end

"""
    rho_v_sat(Tk, phPa)

Saturation vapor density over liquid water [kg/m³], ρ_v* = e*/(R_v T) with the Buck
(1981) saturation vapor pressure (pressure-enhanced, e* converted from hPa to Pa).
For saturated air this equals `rho_d * q_sat_liquid(Tk, phPa)` identically.
"""
function rho_v_sat(Tk::Float64, phPa::Float64)

    return 100.0 * sat_pressure_liquid_buck(Tk, phPa) / (Rv * Tk)
end

"""
    internal_energy_bf02(Tk, q_v, q_l)

Moist internal energy per unit mass of dry air [J/kg] in the Bryan & Fritsch (2002)
convention: e_i = (C_vd + q_v C_vv + q_l C_pv) T − L_v(T) q_l. Energies are referenced
to vapor, so liquid carries u_l = C_pv T − L_v(T) (negative); the internal energy of
vaporization is u_v − u_l = L_v − R_v T.
"""
function internal_energy_bf02(Tk::Float64, q_v::Float64, q_l::Float64)

    return ((Cvd + (q_v * Cvv) + (q_l * Cpv)) * Tk) - (L_v(Tk) * q_l)
end

"""
    dry_density(xi)

Dry-air density [kg/m³] from the log-density variable `xi = ln(rho_d/rho_d0)`.
Inverse of [`log_dry_density`](@ref).
"""
function dry_density(xi::Float64)

    return rho_d0 * exp(xi)
end

"""
    log_dry_density(rho_d)

Log-density variable `xi = ln(rho_d/rho_d0)` from dry-air density [kg/m³].
Inverse of [`dry_density`](@ref).
"""
function log_dry_density(rho_d::Float64)

    return log(rho_d/rho_d0)
end

"""
    P_s(Tk, rho_d, q_v)

Partial derivative of pressure with respect to entropy, `∂p/∂s`, at constant dry-air
density and vapor content [hPa·K/J]. (Holding `rho_d` and `q_v` fixed is equivalent to
holding `rho_d` and the partial density `rho_v = q_v·rho_d` fixed.)
"""
function P_s(Tk::Float64, rho_d::Float64, q_v::Float64)

    Cfactor = Cvd + (q_v * Cvv)
    return Tk * ((rho_d * Rd) + (q_v * rho_d * Rv)) / Cfactor
end

"""
    P_xi(Tk, rho_d, q_v)

Partial derivative of pressure with respect to the log-density `ξ = ln(rho_d/rho_d0)`,
`∂p/∂ξ`, at constant entropy and vapor mixing ratio [Pa]. (`ξ` is a physical change of
variable here, independent of any prognostic-variable transform.)
"""
function P_xi(Tk::Float64, rho_d::Float64, q_v::Float64)

    return (Rd + (q_v * rho_d * Rv)) * ((rho_d * Tk) + P_s(Tk, rho_d, q_v))
end

"""
    P_qv(Tk, rho_d, q_v)

Partial derivative of pressure with respect to water vapor mixing ratio, `∂p/∂q_v`, at
constant entropy and dry-air density [Pa]. Zero when `q_v` is zero.
"""
function P_qv(Tk::Float64, rho_d::Float64, q_v::Float64)

    if (q_v != 0.0)
        rho_v = q_v * rho_d
        qfactor = Rv * (1 + log(rho_v/rho_v0)) - (Cvv * log(Tk/T_0)) - L_v(T_0)/T_0
        qfactor *= P_s(Tk, rho_d, q_v)
        return (rho_d * Rv * Tk) + qfactor
    else
        return 0.0
    end
end

"""
    P_rhod(Tk, rho_d, q_v)

Partial derivative of pressure with respect to dry-air density at constant entropy and
vapor mixing ratio, `∂p/∂rho_d = P_xi/rho_d` [Pa/(kg/m³)].
"""
P_rhod(Tk::Float64, rho_d::Float64, q_v::Float64) = P_xi(Tk, rho_d, q_v) / rho_d

"""
    P_rhov(Tk, rho_d, q_v)

Partial derivative of pressure with respect to the vapor partial density
`rho_v = q_v·rho_d` at constant entropy and dry-air density, `∂p/∂rho_v = P_qv/rho_d`
[Pa/(kg/m³)].
"""
P_rhov(Tk::Float64, rho_d::Float64, q_v::Float64) = P_qv(Tk, rho_d, q_v) / rho_d

"""
    potential_temperature(s, rho_d, q_v)

Dry potential temperature [K] from physical state `(s, rho_d, q_v)`.
"""
function potential_temperature(s::Float64, rho_d::Float64, q_v::Float64)

    Tk = temperature(s, rho_d, q_v)
    p = pressure(s, rho_d, q_v)
    theta = Tk * (p_0 / p)^(Rd/Cpd)
end

"""
    reversible_theta_e(s, rho_d, q_v, q_l=0.0)

Reversible equivalent potential temperature [K] from physical state, accounting for
vapor and liquid water content.

# References
- Emanuel, K. A. (1994). *Atmospheric Convection*. Oxford University Press.
"""
function reversible_theta_e(s::Float64, rho_d::Float64, q_v::Float64, q_l::Float64 = 0.0)

    Tk = temperature(s, rho_d, q_v)
    p = pressure(s, rho_d, q_v)
    q_t = q_v + q_l
    e = vapor_pressure(p, q_v)
    es = sat_pressure_liquid_buck(Tk, p)
    theta_term = Tk * (p_0 / (p-e))^(Rd/(Cpd + (Cl * q_t)))
    H_term = (e/es)^((-Rv * q_v)/(Cpd + (Cl * q_t)))
    exp_term = exp(L_v(Tk) * q_v / ((Cpd + (Cl * q_t)) * Tk))
    return theta_term * H_term * exp_term
end

"""
    theta_rho(s, rho_d, q_v, q_l=0.0)

Density potential temperature [K] from physical state, accounting for the effect of
water vapor and liquid water on air density (virtual temperature effect).
"""
function theta_rho(s::Float64, rho_d::Float64, q_v::Float64, q_l::Float64 = 0.0)

    q_t = q_v + q_l
    theta = potential_temperature(s, rho_d, q_v)
    return theta * (1.0 + (q_v / Eps)) / (1.0 + q_t)
end

end # module Thermodynamics
