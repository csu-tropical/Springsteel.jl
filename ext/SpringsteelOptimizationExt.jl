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
    info = Dict{String, Any}()

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
    grid.spectral[:, 1] .= a

    # Compute physical values
    var = get(prob.parameters, "var", "")
    M_eval = Springsteel.assemble_operator(
        grid, [Springsteel.OperatorTerm(0, 0, 0, nothing)], var)
    phys = M_eval * a

    converged = (opt_sol.retcode == Optimization.SciMLBase.ReturnCode.Success)
    info["retcode"] = string(opt_sol.retcode)
    info["minimum"] = opt_sol.objective

    return Springsteel.SpringsteelSolution(grid, a, phys, converged, info)
end

end # module
