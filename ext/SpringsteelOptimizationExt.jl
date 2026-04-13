module SpringsteelOptimizationExt

using Springsteel
using Optimization

function Springsteel.solve(prob::SpringsteelProblem{OptimizationBackend})
    if prob.cost === nothing
        throw(ArgumentError("OptimizationBackend requires a cost function"))
    end

    grid = prob.grid
    backend = prob.backend
    cost_fn = prob.cost

    # Initial guess: current spectral coefficients (flattened)
    u0 = copy(grid.spectral[:, 1])

    # Build optimization function
    # The cost functional maps spectral coefficients to a scalar
    function opt_cost(u, p)
        var = get(prob.parameters, "var", "")
        M_eval = Springsteel.assemble_operator(
            grid, [Springsteel.OperatorTerm(0, 0, 0, nothing)], var)
        phys = M_eval * u
        return cost_fn(phys, prob.parameters)
    end

    opt_f = OptimizationFunction(opt_cost)
    opt_prob = OptimizationProblem(opt_f, u0, nothing)

    # Get algorithm - must be passed as an actual Optimization algorithm object
    alg = backend.options["algorithm_object"]

    # Solve
    maxiters = get(backend.options, "maxiters", 1000)
    opt_sol = Optimization.solve(opt_prob, alg; maxiters=maxiters)

    # Extract solution
    a = opt_sol.u

    # Compute physical values
    var = get(prob.parameters, "var", "")
    var_name = isempty(var) ? first(keys(grid.params.vars)) : var
    M_eval = Springsteel.assemble_operator(
        grid, [Springsteel.OperatorTerm(0, 0, 0, nothing)], var)
    phys = M_eval * a

    converged = (opt_sol.retcode == Optimization.SciMLBase.ReturnCode.Success)

    # Narrow the source grid to the solved variable and write results into
    # the copy (leaves `grid` untouched — non-mutating contract of `solve`).
    sol_grid = Springsteel._subgrid_for_solution(grid, [var_name])
    @inbounds for r in 1:length(phys)
        sol_grid.physical[r, 1, 1] = phys[r]
    end
    @inbounds for r in 1:length(a)
        sol_grid.spectral[r, 1] = a[r]
    end

    return Springsteel.SpringsteelSolution(sol_grid, 1, converged)
end

end # module
