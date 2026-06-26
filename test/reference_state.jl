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

    @testset "reference_column dispatch" begin
        # natural_column on a Chebyshev column returns an R0 Chebyshev1D
        nc = Springsteel.natural_column(column, (kMin=0.0, kMax=10000.0, kDim=64, b_kDim=64))
        @test nc isa Chebyshev.Chebyshev1D
    end

    rm(dry_file; force=true)
    rm(moist_file; force=true)
end
