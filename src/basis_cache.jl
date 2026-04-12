"""
    clear_basis_caches!()

Clear all cached basis templates (Spline1D, Fourier1D, Chebyshev1D).

Subsequent basis construction will rebuild from scratch. Use this to reclaim
memory in long-running processes that have accumulated many distinct parameter
sets, or in tests that need cold-cache behavior.
"""
function clear_basis_caches!()
    CubicBSpline._clear_spline_cache!()
    Fourier._clear_fourier_cache!()
    Chebyshev._clear_chebyshev_cache!()
    return nothing
end

"""
    basis_cache_sizes() -> NamedTuple

Return the number of cached templates for each basis type.
"""
function basis_cache_sizes()
    return (spline    = CubicBSpline._spline_cache_size(),
            fourier   = Fourier._fourier_cache_size(),
            chebyshev = Chebyshev._chebyshev_cache_size())
end
