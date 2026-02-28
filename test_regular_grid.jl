using Springsteel

#──────────────────────────────────────────────────────────────────────────────
# Test RLZ regularGridTransform at mish r,z points vs gridTransform! output
#──────────────────────────────────────────────────────────────────────────────
gp = SpringsteelGridParameters(geometry="RLZ",
    iMin=0.0, iMax=80.0, num_cells=6, kMin=0.0, kMax=20.0, kDim=10,
    vars=Dict("u"=>1),
    BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
    BCB=Dict("u"=>Chebyshev.R0), BCT=Dict("u"=>Chebyshev.R0))
grid = createGrid(gp)
σ_r = 25.0; σ_z = 5.0; z0 = 10.0

begin
    idx = 1
    for r in 1:grid.params.iDim
        ri = r + grid.params.patchOffsetL; lpts = 4 + 4*ri
        rv = grid.ibasis.data[1,1].mishPoints[r]
        for l in 1:lpts
            for z in 1:grid.params.kDim
                zv = grid.kbasis.data[1].mishPoints[z]
                grid.physical[idx,1,1] = exp(-(rv/σ_r)^2 - ((zv-z0)/σ_z)^2)
                idx += 1
            end
        end
    end
end

spectralTransform!(grid)
gridTransform!(grid)

# Evaluate regularGridTransform at mish r,z points
r_mish = collect(Float64, grid.ibasis.data[1,1].mishPoints)
z_mish = collect(Float64, grid.kbasis.data[1].mishPoints)
λ_single = [0.1]   # single λ point (axisymmetric function)

reg = regularGridTransform(grid, r_mish, λ_single, z_mish)
n_r = length(r_mish)
n_z = length(z_mish)

# Compare against gridTransform! output
max_err_vs_phys = 0.0
max_err_vs_exact = 0.0
begin
    gidx = 1
    for ri in 1:n_r
        ri_abs = ri + grid.params.patchOffsetL
        lpts_r = 4 + 4*ri_abs
        rv = r_mish[ri]
        for zi in 1:n_z
            flat_reg = (ri-1)*1*n_z + zi
            f_reg   = reg[flat_reg, 1, 1]
            f_phys  = grid.physical[gidx, 1, 1]   # first l point of ring ri
            zv = z_mish[zi]
            f_exact = exp(-(rv/σ_r)^2 - ((zv-z0)/σ_z)^2)
            max_err_vs_phys  = max(max_err_vs_phys,  abs(f_reg - f_phys))
            max_err_vs_exact = max(max_err_vs_exact, abs(f_reg - f_exact))
            gidx += (zi == n_z) ? (lpts_r - 1)*n_z + 1 : 1
        end
    end
end

println("RLZ: max |reg - gridTransform!| at mish pts = ", max_err_vs_phys)
println("RLZ: max |reg - exact| at mish pts          = ", max_err_vs_exact)

#──────────────────────────────────────────────────────────────────────────────
# Test RL non-axisymmetric: use cosine function f(r,λ) = r*cos(λ)
#──────────────────────────────────────────────────────────────────────────────
gp_rl = SpringsteelGridParameters(geometry="RL", iMin=0.0, iMax=100.0, num_cells=10,
    vars=Dict("u"=>1), BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0))
grid_rl = createGrid(gp_rl)

begin
    g = 1
    for r in 1:grid_rl.params.iDim
        ri = r + grid_rl.params.patchOffsetL; lpts = 4 + 4*ri
        rv = grid_rl.ibasis.data[1,1].mishPoints[r]
        for l in 1:lpts
            lv = grid_rl.jbasis.data[r,1].mishPoints[l]
            grid_rl.physical[g,1,1] = rv * cos(lv)   # f = r*cos(λ)
            g += 1
        end
    end
end
spectralTransform!(grid_rl)
gridTransform!(grid_rl)

# Evaluate at regular grid including non-trivial λ values
n_r_out = 5; n_λ_out = 16
r_out = collect(LinRange(5.0, 95.0, n_r_out))
λ_out = [2π*j/n_λ_out for j in 0:n_λ_out-1]
reg_rl = regularGridTransform(grid_rl, r_out, λ_out)

max_err_rl = 0.0
for ri in 1:n_r_out, ji in 1:n_λ_out
    rv = r_out[ri]; lv = λ_out[ji]
    f_expected = rv * cos(lv)
    f_got = reg_rl[(ri-1)*n_λ_out + ji, 1, 1]
    max_err_rl = max(max_err_rl, abs(f_got - f_expected))
end
println("\nRL non-axisymmetric f=r*cos(λ): max error = ", max_err_rl)
println("(sample: r=50,λ=π/4: expected=$(round(50*cos(π/4),digits=4)), got=$(round(reg_rl[(3-1)*n_λ_out+3,1,1],digits=4)))")
