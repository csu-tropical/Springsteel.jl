@testset "Reference state" begin

    using Springsteel.Thermodynamics
    import Springsteel: ref_entropy, ref_rho_d, ref_rho_v, ref_rho_c, ref_sat,
        sound_speed_sq, reference_temperature, reference_column

    # A vertical Chebyshev column with natural (R0) boundaries, 0–10 km
    cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=10000.0, zDim=64, bDim=64,
        BCB=Chebyshev.R0, BCT=Chebyshev.R0)
    column = Chebyshev.Chebyshev1D(cp)
    z = column.mishPoints

    # Write a sounding file (surface line `p theta q_v`, then `z theta q_v`; q_v g/kg)
    function write_sounding(path, thetafun, qvfun)
        open(path, "w") do io
            println(io, "1000.0 $(thetafun(0.0)) $(qvfun(0.0))")
            for zi in 500.0:500.0:10000.0
                println(io, "$(zi) $(thetafun(zi)) $(qvfun(zi))")
            end
        end
    end

    dry_file = tempname()
    write_sounding(dry_file, z -> 300.0, z -> 0.0)               # dry neutral
    moist_file = tempname()
    write_sounding(moist_file, z -> 300.0, z -> 10.0)            # 10 g/kg uniform vapor

    # Hydrostatic residual felt by the dynamics, from stored physical profiles:
    #   res = P_s*s_z + P_xi*xi_z + P_qv*q_v_z + g*rho_d*(1+q_v+q_l)
    # with xi_z = rho_d_z/rho_d. Normalised by g*rho_t.
    function max_rel_residual(rs)
        s  = ref_entropy(rs)[:, 1];  s_z = ref_entropy(rs)[:, 2]
        rd = ref_rho_d(rs)[:, 1];   rd_z = ref_rho_d(rs)[:, 2]
        rv = ref_rho_v(rs)
        if rv === 0.0
            q_v = zero(rd); q_v_z = zero(rd)
        else
            q_v = rv[:, 1] ./ rd
            q_v_z = (rv[:, 2] .- q_v .* rd_z) ./ rd     # (rho_v_z - q_v*rho_d_z)/rho_d
        end
        rc = ref_rho_c(rs)
        q_l = rc === 0.0 ? zero(rd) : rc[:, 1] ./ rd
        Tk = temperature.(s, rd, q_v)
        xi_z = rd_z ./ rd
        res = P_s.(Tk, rd, q_v) .* s_z .+ P_xi.(Tk, rd, q_v) .* xi_z .+
              P_qv.(Tk, rd, q_v) .* q_v_z .+ gravity .* rd .* (1.0 .+ q_v .+ q_l)
        rho_t = rd .* (1.0 .+ q_v .+ q_l)
        # Interior points only (spectral derivatives are least accurate at the walls)
        n = length(rd); interior = 4:(n-3)
        return maximum(abs.(res[interior])) / maximum(gravity .* rho_t)
    end

    @testset "Dry reference (moisture=false)" begin
        rs = Springsteel.calculate_reference_state(dry_file, z, column; moisture=false)
        @test rs isa DryReferenceState
        @test all(ref_rho_d(rs)[:, 1] .> 0.0)
        @test ref_rho_v(rs) === 0.0
        @test ref_rho_c(rs) === 0.0
        @test sound_speed_sq(rs) > 0.0
        @test sqrt(sound_speed_sq(rs)) > 250.0 && sqrt(sound_speed_sq(rs)) < 400.0
        # Surface dry-air density ~ p/(Rd T): 1000 hPa, theta=300 => T0=300
        @test ref_rho_d(rs)[1, 1] ≈ 100_000.0 / (Rd * 300.0) rtol = 1e-3
        @test max_rel_residual(rs) < 1e-3
        # Temperature decreases with height for a neutral (constant-theta) atmosphere
        T = reference_temperature(rs)
        @test T[1] > T[end]
    end

    @testset "Moist reference (moisture=true)" begin
        rs = Springsteel.calculate_reference_state(moist_file, z, column; moisture=true)
        @test rs isa MoistReferenceState
        @test ref_rho_v(rs) isa Matrix{Float64}
        @test ref_rho_c(rs) === 0.0
        @test all(ref_rho_v(rs)[:, 1] .> 0.0)
        # rho_v = rho_d * q_v with q_v ~ 0.01
        q_v = ref_rho_v(rs)[:, 1] ./ ref_rho_d(rs)[:, 1]
        @test all(q_v .> 0.0)
        @test q_v[1] ≈ 0.010 rtol = 0.05
        @test all(ref_sat(rs)[:, 1] .> 0.0)        # subsaturated => 0 < q_v/q_sat < 1
        @test max_rel_residual(rs) < 1e-3
    end

    @testset "interpolate_reference_state" begin
        rs = Springsteel.interpolate_reference_state(moist_file, z, column; moisture=true)
        @test rs isa MoistReferenceState
        @test all(ref_rho_d(rs)[:, 1] .> 0.0)
        @test sound_speed_sq(rs) > 0.0
    end

    @testset "exact_reference_state (physical format)" begin
        # Write a pre-balanced physical file (z s rho_d rho_v rho_c) with nonzero
        # condensate, read it back, and confirm the profiles and condensate carry through.
        exact_file = tempname()
        s_in = [entropy(300.0 - 5e-3 * zi, 1.1, 0.01) for zi in z]
        rho_d_in = [1.15 - 5e-5 * zi for zi in z]
        rho_v_in = 0.01 .* rho_d_in
        rho_c_in = 0.001 .* rho_d_in        # uniform saturated cloud loading
        open(exact_file, "w") do io
            for i in eachindex(z)
                println(io, "$(z[i]) $(s_in[i]) $(rho_d_in[i]) $(rho_v_in[i]) $(rho_c_in[i])")
            end
        end
        rs = Springsteel.exact_reference_state(exact_file, z, column)
        @test rs isa CondensateReferenceState
        @test ref_rho_c(rs) isa Matrix{Float64}
        @test isapprox(ref_rho_d(rs)[:, 1], rho_d_in; rtol=1e-10)
        @test isapprox(ref_rho_v(rs)[:, 1], rho_v_in; rtol=1e-10)
        @test isapprox(ref_rho_c(rs)[:, 1], rho_c_in; rtol=1e-10)
        @test all(ref_rho_c(rs)[:, 1] .> 0.0)
        rm(exact_file; force=true)
    end

    @testset "reference_column dispatch" begin
        # natural_column on a Chebyshev column returns an R0 Chebyshev1D
        nc = Springsteel.natural_column(column, (kMin=0.0, kMax=10000.0, kDim=64, b_kDim=64))
        @test nc isa Chebyshev.Chebyshev1D
    end

    @testset "PressureReferenceState" begin
        import Springsteel: ref_pressure, ref_rho_t, ref_total_energy, ref_qss

        # -- exact builder: dry neutral column (analytic Exner hydrostatic) --
        theta0 = 300.0
        exner = @. 1.0 - gravity * z / (Cpd * theta0)
        Tk_dry = theta0 .* exner
        p_dry = @. 100000.0 * exner^(Cpd / Rd)          # Pa
        rho_d_dry = p_dry ./ (Rd .* Tk_dry)
        pfile = tempname()
        open(pfile, "w") do io
            for i in eachindex(z)
                println(io, "$(z[i]) $(p_dry[i]) $(rho_d_dry[i]) 0.0 0.0")
            end
        end
        rs = Springsteel.exact_pressure_reference_state(pfile, z, column)
        @test rs isa PressureReferenceState
        @test ref_pressure(rs) isa Matrix{Float64}
        @test isapprox(ref_pressure(rs)[:, 1], p_dry; rtol=1e-8)
        @test isapprox(ref_rho_t(rs)[:, 1], rho_d_dry; rtol=1e-8)
        @test isapprox(reference_temperature(rs), Tk_dry; rtol=1e-8)   # EOS temperature
        # Hydrostatic: dp/dz = -g*rho_t (interior; construction is analytic)
        n = length(z); interior = 4:(n - 3)
        res = ref_pressure(rs)[:, 2] .+ gravity .* ref_rho_t(rs)[:, 1]
        @test maximum(abs.(res[interior])) / (gravity * maximum(rho_d_dry)) < 1e-6
        # Dry air: Q_ssbar = -rho_v_sat < 0; E_t = rho_d*Cvd*T + rho_t*g*z
        @test all(ref_qss(rs)[:, 1] .< 0.0)
        E_expected = rho_d_dry .* (Cvd .* Tk_dry .+ gravity .* z)
        @test isapprox(ref_total_energy(rs)[:, 1], E_expected; rtol=1e-8)
        @test 250.0 < sqrt(sound_speed_sq(rs)) < 400.0
        rm(pfile; force=true)

        # -- exact builder: saturated cloudy column (Q_ssbar = 0 identically) --
        Tk_m = @. 290.0 - 0.005 * z
        p_m = @. 90000.0 * exp(-z / 8000.0)             # Pa (need not be hydrostatic here)
        rho_v_m = rho_v_sat.(Tk_m, p_m ./ 100.0)
        rho_d_m = (p_m .- (Rv .* Tk_m .* rho_v_m)) ./ (Rd .* Tk_m)
        rho_c_m = 1.0e-3 .* rho_d_m
        pfile2 = tempname()
        open(pfile2, "w") do io
            for i in eachindex(z)
                println(io, "$(z[i]) $(p_m[i]) $(rho_d_m[i]) $(rho_v_m[i]) $(rho_c_m[i])")
            end
        end
        rs2 = Springsteel.exact_pressure_reference_state(pfile2, z, column)
        @test isapprox(reference_temperature(rs2), Tk_m; rtol=1e-8)
        @test maximum(abs.(ref_qss(rs2)[:, 1])) < 1e-10    # saturated: Q_ss = 0 pointwise
        @test isapprox(ref_rho_c(rs2)[:, 1], rho_c_m; rtol=1e-6)
        q_v_m = rho_v_m ./ rho_d_m; q_l_m = rho_c_m ./ rho_d_m
        E2 = rho_d_m .* internal_energy_bf02.(Tk_m, q_v_m, q_l_m) .+
             (rho_d_m .+ rho_v_m .+ rho_c_m) .* gravity .* z
        @test isapprox(ref_total_energy(rs2)[:, 1], E2; rtol=1e-8)
        rm(pfile2; force=true)

        # -- sounding builder: hydrostatic + monotone p --
        rs3 = Springsteel.calculate_pressure_reference_state(moist_file, z, column)
        @test rs3 isa PressureReferenceState
        p3 = ref_pressure(rs3)[:, 1]
        @test all(diff(p3) .< 0.0)                       # monotone decreasing
        res3 = ref_pressure(rs3)[:, 2] .+ gravity .* ref_rho_t(rs3)[:, 1]
        @test maximum(abs.(res3[interior])) / (gravity * maximum(ref_rho_t(rs3)[:, 1])) < 1e-3
        # 10 g/kg is subsaturated at low levels but supersaturated in the cold upper
        # column (constant-q_v fixture sounding), so only check the lowest levels
        @test all(ref_qss(rs3)[1:8, 1] .< 0.0)
        T3 = reference_temperature(rs3)
        @test T3[1] > T3[end]
    end

    rm(dry_file; force=true)
    rm(moist_file; force=true)
end
