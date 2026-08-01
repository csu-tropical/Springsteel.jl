```@meta
CurrentModule = Springsteel
```

# Thermodynamics and Reference States

Springsteel ships two closely related pieces that atmospheric clients share: a
basis-agnostic `Thermodynamics` submodule (physical equation of state and diagnostics),
and a hydrostatic **reference state** module that fits a balanced base-state profile onto
a Springsteel vertical column.

Neither is loaded into your namespace by `using Springsteel` alone. The thermodynamics
submodule is imported explicitly, and the reference-state accessors are imported by name:

```julia
using Springsteel
using Springsteel.Thermodynamics          # constants and diagnostics
import Springsteel: ref_pressure, ref_rho_d, ref_sigma   # accessors, as needed
```

Only the reference-state **types** are exported: `AbstractReferenceState`,
`DryReferenceState`, `MoistReferenceState`, `CondensateReferenceState`, and
`PressureReferenceState`.

---

## Reference-state types

A reference state stores one or more profiles, each an `(nlevels, 3)` array holding the
value and its first two vertical derivatives — the same derivative-slot layout the grid
transforms use, so a base state can be differentiated without a further transform.

| Type | Prognostic variable it supports | Carries |
|:---|:---|:---|
| `DryReferenceState` | Dry entropy / log-density sets | `sbar`, `rho_dbar`, `sigmabar` |
| `MoistReferenceState` | Moist entropy sets | adds `rho_vbar`, `satbar` |
| `CondensateReferenceState` | Moist sets with condensate loading | adds `rho_cbar` |
| `PressureReferenceState` | **Total-energy** sets carrying `p` prognostically | `pbar` [Pa], partial and total densities, `E_tbar`, `Q_ssbar` |

```@docs
AbstractReferenceState
DryReferenceState
MoistReferenceState
CondensateReferenceState
PressureReferenceState
```

### Accessors

```@docs
Springsteel.ref_entropy
Springsteel.ref_rho_d
Springsteel.ref_rho_v
Springsteel.ref_rho_c
Springsteel.ref_sat
Springsteel.ref_sigma
Springsteel.ref_pressure
Springsteel.ref_rho_t
Springsteel.ref_total_energy
Springsteel.ref_qss
```

`ref_rho_v`, `ref_rho_c`, and `ref_sat` return `0.0` on states that do not carry the
corresponding profile, so equation sets can be written once and run dry or moist without
branching.

### The entropy-density profile

`sigmabar` is $\hat\sigma = \hat\rho_d \hat s$, the entropy density, with its vertical
derivatives taken spectrally rather than formed from a product rule on the stored
derivative slots. It is present on all of the entropy-based states and is what an
entropy-density (σ) equation set advances.

---

## Builders

Two families, differing in what the input file is trusted to provide.

**Sounding path** — read a sounding, interpolate it to the column's levels, and build a
balanced state by spectral integration:

```@docs
Springsteel.calculate_reference_state
Springsteel.calculate_pressure_reference_state
Springsteel.interpolate_reference_state
```

**Exact path** — read a physical-format file that already contains the profiles, and use
them directly (the file's own pressure becomes the anchor and an accuracy check):

```@docs
Springsteel.exact_reference_state
Springsteel.exact_pressure_reference_state
```

### Hydrostatic balance

Pass `hydrostatic = true` to the pressure builders to enforce **discrete** hydrostatic
balance, `dp/dz = -g\rho_t`, rather than merely the analytic relation:

```julia
column = Spline1D(SplineParameters(xmin = 0.0, xmax = 25.0e3, num_cells = 50, mubar = 3))
rs = calculate_pressure_reference_state("dunion_MT.snd", z, column; hydrostatic = true)
```

Two things make this exact rather than approximate, and both matter:

1. **The fixed-point sweep is iterated to convergence.** The sweep *oscillates* before it
   settles, so a fixed small iteration count stops it mid-swing. On a 50-cell / 25 km
   tropical-cyclone grid with the Dunion moist-tropical sounding, the lid pressure runs
   1912 → 4010 → 1512 → 3550 → 2273 → … → 2707 Pa. It now sweeps to `1e-12` or throws.

2. **The pressure antiderivative is kept, not re-fitted.** Pressure is built by
   integrating the fitted $-g\rho_t$; re-fitting that result's *values* costs ~0.03%,
   which on $10^5$ Pa is ~30 Pa, or ~0.06 Pa/m across a 500 m cell — 10–17% of $g\rho_t$
   where $p$ is small. The antiderivative is used for the value slot and the derivative
   slots are snapped to $-g\bar\rho_t$, making the balance exact. Step 1 is what makes
   that snap a 0.03% adjustment instead of a 17% one.

With both in place, $-(\mathrm{d}p/\mathrm{d}z + g\rho_t)/\rho_t$ is 0.0 at every level.

Because the same entry point serves both the sounding and exact builders, a values-only
`.ref` round trip also carries a balanced reference, with no file-format change.

!!! note "Integration gauge"
    The reference-state integrations use the generic `IInttransform`, which anchors the
    antiderivative so it equals `C0` at `xmin` — "`C0` is the value at the bottom". The
    spline-native `SIInttransform` uses the Ooyama (2002) gauge instead (zero near the
    domain centre) and is not interchangeable. See the
    [CubicBSpline](cubicbspline.md#Generic-Wrappers) page.

---

## Thermodynamics submodule

```julia
using Springsteel.Thermodynamics
```

Constants (`Rd`, `Rv`, `Cpd`, `Cvd`, `gravity`, `L_v0`, `p_0`, …) and diagnostics over
physical variables. The submodule is basis-agnostic — it knows nothing about grids — so it
can be used standalone.

Saturation and moisture: `sat_pressure_liquid`, `sat_pressure_ice`,
`sat_pressure_liquid_buck` (and its `_dT` derivative), `sat_pressure_ice_buck`,
`q_sat_liquid`, `q_sat_ice`, `rho_v_sat`, `mixing_ratio`, `dewpoint`, `L_v`.

State and energy: `temperature`, `pressure`, `vapor_pressure`, `dry_density`,
`log_dry_density`, `entropy`, `vapor_entropy`, `internal_energy_bf02`.

Potential temperatures: `potential_temperature`, `reversible_theta_e`, `theta_rho`.

Partial derivatives of entropy used by the semi-implicit solves: `P_s`, `P_xi`, `P_qv`,
`P_rhod`, `P_rhov`.

### Two ways to get θ

The three-argument method `potential_temperature(s, rho_d, q_v)` inverts entropy. The
two-argument method `potential_temperature(p_Pa, rho_d)` gives the same quantity directly
from the $(p, \rho_d)$ pair,

```math
\theta_d \equiv \frac{p_0^{\kappa}}{R_d}\, \frac{p^{1-\kappa}}{\rho_d}
```

which is what a total-energy equation set wants: it carries pressure prognostically, so it
can diagnose θ with no entropy inversion and no spectral transform.

!!! warning "Units"
    `potential_temperature(p_Pa, rho_d)` takes pressure in **pascals**, matching
    `PressureReferenceState.pbar`. The saturation functions take **hPa**.

### Total-energy helpers

`internal_energy_bf02` is the Bannon (2002) internal energy used in the total energy
density $E_t = \rho_d e_i + \rho_t g z$ that `PressureReferenceState` stores, and
`rho_v_sat` gives the saturation vapour density behind its supersaturation profile
$Q_{ss} = \rho_v - \rho_{v,sat}(T, p)$.

### Thermodynamics API reference

```@autodocs
Modules = [Springsteel.Thermodynamics]
Order   = [:module, :function]
```

---

## See also

- [CubicBSpline](cubicbspline.md) — the spline column a reference state is fitted on,
  and the `IInttransform` gauge
- [Boundary Conditions](boundary_conditions.md) — `R1T1X`, the wall condition a
  hydrostatically-balanced compressible model needs at its lower boundary
- [SpringsteelGrid](springsteel_grid.md) — grid construction and parameters
