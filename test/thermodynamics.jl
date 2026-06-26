@testset "Thermodynamics" begin

    using Springsteel.Thermodynamics

    @testset "Constants consistency" begin
        @test Cpd ≈ Cvd + Rd
        @test Cpv ≈ Cvv + Rv
        @test Eps ≈ Rd / Rv
        @test rho_d0 ≈ 100.0 * p_0 / (T_0 * Rd)
        @test rho_v0 ≈ 100.0 * sat_pressure_liquid(T_0) / (T_0 * Rv)
    end

    @testset "Saturation vapor pressure" begin
        # Bolton (1980) anchor values
        @test sat_pressure_liquid(273.16) ≈ 6.116436706236274
        @test sat_pressure_liquid(300.0) ≈ 35.34519666889136
        # Buck >= small positive, monotone increasing in T
        @test sat_pressure_liquid_buck(300.0, 1000.0) > sat_pressure_liquid_buck(290.0, 1000.0)
        @test sat_pressure_ice_buck(263.0, 800.0) > 0.0
        # Buck dT matches a finite difference
        Tk, p = 295.0, 950.0
        dfd = (sat_pressure_liquid_buck(Tk + 1e-4, p) - sat_pressure_liquid_buck(Tk - 1e-4, p)) / 2e-4
        @test sat_pressure_liquid_buck_dT(Tk, p) ≈ dfd rtol=1e-5
        # Saturation mixing ratio round-trips through vapor pressure
        q = q_sat_liquid(295.0, 950.0)
        @test q > 0.0
        @test mixing_ratio(950.0, sat_pressure_liquid_buck(295.0, 950.0)) ≈ q
    end

    @testset "Latent heat" begin
        @test L_v(273.16) ≈ 2.501e6
        @test L_v(300.0) ≈ L_v0 + (Cpv - Cl) * (300.0 - T_0)
    end

    @testset "Entropy <-> temperature round trip" begin
        for (Tk, rho_d, q_v) in ((300.0, 1.0, 0.01), (280.0, 1.1, 0.0),
                                 (250.0, 0.7, 0.002), (305.0, 1.15, 0.018))
            s = entropy(Tk, rho_d, q_v)
            @test temperature(s, rho_d, q_v) ≈ Tk
        end
    end

    @testset "Pressure / ideal-gas consistency" begin
        s = entropy(300.0, 1.0, 0.01)
        p = pressure(s, 1.0, 0.01)
        Tk = temperature(s, 1.0, 0.01)
        @test p ≈ 0.01 * (Rd + 0.01 * Rv) * Tk * 1.0
        # vapor pressure / mixing ratio inverse pair
        e = vapor_pressure(p, 0.01)
        @test mixing_ratio(p, e) ≈ 0.01
        # dewpoint <= temperature for subsaturated air
        @test dewpoint(p, 0.01) <= Tk
        # dry-air edge case (q_v = 0)
        sd = entropy(290.0, 1.05, 0.0)
        @test pressure(sd, 1.05, 0.0) ≈ 0.01 * Rd * temperature(sd, 1.05, 0.0) * 1.05
    end

    @testset "dry_density <-> log_dry_density" begin
        for rho_d in (0.8, 1.0, 1.2)
            @test dry_density(log_dry_density(rho_d)) ≈ rho_d
        end
        @test dry_density(0.0) ≈ rho_d0
    end

    @testset "P_s vs finite difference" begin
        # P_s = ∂p/∂s with pressure in Pa; `pressure` returns hPa, so scale the FD by 100.
        rho_d, q_v = 1.0, 0.01
        s0 = entropy(300.0, rho_d, q_v)
        Tk = temperature(s0, rho_d, q_v)
        dfd = 100.0 * (pressure(s0 + 1e-3, rho_d, q_v) - pressure(s0 - 1e-3, rho_d, q_v)) / 2e-3
        @test P_s(Tk, rho_d, q_v) ≈ dfd rtol=1e-4
    end

    @testset "Diagnostics" begin
        s, rho_d, q_v = entropy(300.0, 1.0, 0.01), 1.0, 0.01
        theta = potential_temperature(s, rho_d, q_v)
        @test theta > temperature(s, rho_d, q_v)        # surface-ish, theta >= T below ~1000 hPa
        # theta_rho exceeds theta when vapor present and no liquid
        @test theta_rho(s, rho_d, q_v, 0.0) > theta
        # liquid loading lowers theta_rho
        @test theta_rho(s, rho_d, q_v, 0.005) < theta_rho(s, rho_d, q_v, 0.0)
        # reversible theta_e is finite and positive
        @test isfinite(reversible_theta_e(s, rho_d, q_v, 0.0))
        @test reversible_theta_e(s, rho_d, q_v, 0.0) > 0.0
    end
end
