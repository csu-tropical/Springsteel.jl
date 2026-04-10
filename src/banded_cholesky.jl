# ============================================================================
# Banded LLᵀ Cholesky factorization for cubic B-spline (P+Q) systems
# ============================================================================
#
# The Ooyama (2002) variational matrix `(P + Q)` for cubic B-splines has fixed
# half-bandwidth 3 (7-point stencil) by construction. After folding the
# boundary conditions through the projection matrix Γ, the resulting open-form
# matrix `Γ(P+Q)Γᵀ` retains bandwidth 3 for all non-periodic boundary types
# (R0, R1Tx, R2T*x, R3, R3X). The PERIODIC case is the only exception: the
# wraparound rows in Γ couple the corner blocks and break the strict band
# structure. We handle that case via a dense Cholesky fallback.
#
# This module provides:
#
#   AbstractSplineFactor      — abstract supertype dispatched by SAtransform!
#   BandedCholesky3           — banded LLᵀ factor in (4 × n) lower-band storage
#   DenseSplineFactor         — wrapper around LinearAlgebra.Cholesky for the
#                               periodic-BC case
#   cholesky_banded3(A)       — constructor; asserts bandwidth ≤ 3
#   cholesky_dense_factor(A)  — periodic fallback constructor
#   LinearAlgebra.ldiv!(out, F, b) — allocation-free solve for both factor types
#
# The bandwidth is hard-coded to 3 because cubic B-splines have a 7-point
# stencil. Higher-order splines would be a separate basis type.

using LinearAlgebra: Symmetric, Cholesky, cholesky, mul!, ldiv!

# ── Type hierarchy ──────────────────────────────────────────────────────────

"""
    AbstractSplineFactor

Abstract supertype for the factorization stored in [`Spline1D`](@ref) for the
variational `(P+Q)` and `(P¹+Q)` systems. Concrete subtypes:

- [`BandedCholesky3`](@ref) — banded LLᵀ Cholesky for the common (non-periodic)
  case; allocation-free `ldiv!`, ~6.4 KB storage at n=200.
- [`DenseSplineFactor`](@ref) — dense LLᵀ Cholesky for the periodic-BC case
  where the banded structure is broken by corner wraparound.

Both expose the same `LinearAlgebra.ldiv!(out, F, b)` API used by
[`SAtransform!`](@ref) and the integration path.
"""
abstract type AbstractSplineFactor end

"""
    BandedCholesky3 <: AbstractSplineFactor

Banded LLᵀ Cholesky factor for symmetric positive-definite matrices with
half-bandwidth ≤ 3 (cubic B-spline (P+Q) systems).

The factor is stored in **lower-band storage** of shape `(4, n)`:
```
L_band[1, j] = L[j, j]                      (diagonal)
L_band[2, j] = L[j+1, j]                    (1st sub-diagonal)
L_band[3, j] = L[j+2, j]                    (2nd sub-diagonal)
L_band[4, j] = L[j+3, j]                    (3rd sub-diagonal)
```
where `L` satisfies `A = L Lᵀ`.

Construct with [`cholesky_banded3`](@ref). Solve with
`LinearAlgebra.ldiv!(out, F, b)`, which is allocation-free.
"""
struct BandedCholesky3 <: AbstractSplineFactor
    L::Matrix{Float64}   # (4, n) lower-band storage; overwritten in place by the factor
    n::Int
end

"""
    DenseSplineFactor <: AbstractSplineFactor

Dense LLᵀ Cholesky fallback for the periodic-BC case, where Γ wraparound
breaks the banded structure of `Γ(P+Q)Γᵀ`. Wraps a standard
`LinearAlgebra.Cholesky{Float64, Matrix{Float64}}`.

The dense path is allocation-free in `ldiv!` (LAPACK `potrs` writes the
solve in place) and matrices are small in practice; periodic boundary
conditions are an edge case in Springsteel since Fourier basis is preferred
for periodic geometries.
"""
struct DenseSplineFactor <: AbstractSplineFactor
    F::Cholesky{Float64, Matrix{Float64}}
end

# ── Constructors ────────────────────────────────────────────────────────────

"""
    cholesky_banded3(A::Symmetric{Float64, Matrix{Float64}}) -> BandedCholesky3

Compute the banded LLᵀ Cholesky factorization of a symmetric positive-definite
matrix with half-bandwidth ≤ 3.

Asserts that all entries `A[i, j]` with `|i - j| > 3` are zero. If your
matrix violates this (e.g. periodic BC corner couplings), use
[`cholesky_dense_factor`](@ref) instead.

Throws `PosDefException` if the matrix is not positive definite.
"""
function cholesky_banded3(A::Symmetric{Float64, Matrix{Float64}})
    n = size(A, 1)
    _assert_bandwidth_3(A)
    L = zeros(Float64, 4, n)
    # Copy lower triangle (within band) into band storage.
    @inbounds for j in 1:n
        ihi = min(n, j + 3)
        for i in j:ihi
            L[1 + i - j, j] = A[i, j]
        end
    end
    _band_factor3!(L, n)
    return BandedCholesky3(L, n)
end

"""
    cholesky_dense_factor(A::Symmetric{Float64, Matrix{Float64}}) -> DenseSplineFactor

Compute a dense LLᵀ Cholesky factorization wrapped as a `DenseSplineFactor`.
Used as the periodic-BC fallback for `Spline1D`.
"""
function cholesky_dense_factor(A::Symmetric{Float64, Matrix{Float64}})
    return DenseSplineFactor(cholesky(A))
end

# ── Bandwidth assertion (insurance against future BC additions) ─────────────

"""
    _assert_bandwidth_3(A) -> Nothing

Verify that all entries of `A` outside the half-bandwidth-3 band are exactly
zero. The cubic B-spline `(P+Q)` matrix has this structure by construction;
the assertion catches any future regression in `calcGammaBC` or
`calcPQfactor` that would silently violate the banded assumption.
"""
function _assert_bandwidth_3(A::Symmetric{Float64, Matrix{Float64}})
    n = size(A, 1)
    @inbounds for j in 1:n
        for i in (j + 4):n
            if A[i, j] != 0.0
                throw(ArgumentError(
                    "cholesky_banded3 expects half-bandwidth ≤ 3, but A[$i, $j] = " *
                    "$(A[i, j]) is outside the band. Use cholesky_dense_factor for " *
                    "periodic BCs or matrices with broader structure."))
            end
        end
    end
    return nothing
end

# ── Banded factor (in-place over lower-band storage) ────────────────────────

# A = L Lᵀ where L has the same lower bandwidth (3) as A.
#
# For j = 1..n:
#   L[j,j] = sqrt(A[j,j] - Σ_{k=max(1,j-3):j-1} L[j,k]²)
#   For i = j+1..min(n, j+3):
#     L[i,j] = (A[i,j] - Σ_{k=max(1,j-3):j-1} L[i,k] · L[j,k]) / L[j,j]
#
# In band storage (lower):
#   L[j,k] for k<j  is L[1 + j - k, k]
#   L[i,k] for k<j  is L[1 + i - k, k]
#   L[j,j]          is L[1, j]
#   L[i,j] for i>j  is L[1 + i - j, j]
#
# All inner-loop accesses are within the (4, n) band storage, so the
# inner sum is at most 3 multiply-adds.
function _band_factor3!(L::Matrix{Float64}, n::Int)
    @inbounds for j in 1:n
        # Diagonal: L[j,j] = sqrt(A[j,j] - Σ L[j,k]² for k in (j-3..j-1))
        d = L[1, j]
        kstart_diag = max(1, j - 3)
        for k in kstart_diag:(j - 1)
            ljk = L[1 + j - k, k]
            d -= ljk * ljk
        end
        if d <= 0.0
            throw(LinearAlgebra.PosDefException(j))
        end
        ljj = sqrt(d)
        L[1, j] = ljj
        inv_ljj = 1.0 / ljj

        # Sub-diagonal entries L[i,j] for i = j+1 .. min(n, j+3).
        # The inner sum runs over k where both L[i,k] and L[j,k] are in-band,
        # i.e. k ≥ max(i-3, j-3) = i-3 (since i > j). The range i-3..j-1 has
        # j-1-(i-3)+1 = j-i+3 terms — 2 for i=j+1, 1 for i=j+2, 0 for i=j+3.
        ihi = min(n, j + 3)
        for i in (j + 1):ihi
            s = L[1 + i - j, j]
            kstart = max(1, i - 3)
            for k in kstart:(j - 1)
                s -= L[1 + i - k, k] * L[1 + j - k, k]
            end
            L[1 + i - j, j] = s * inv_ljj
        end
    end
    return L
end

# ── Forward / backward banded triangular solves ─────────────────────────────

# Forward solve: L y = b   (writes y in place, reading b separately)
function _band_forward_solve3!(y::AbstractVector{Float64},
                                L::Matrix{Float64},
                                n::Int,
                                b::AbstractVector{Float64})
    @inbounds for i in 1:n
        s = b[i]
        kstart = max(1, i - 3)
        for k in kstart:(i - 1)
            s -= L[1 + i - k, k] * y[k]
        end
        y[i] = s / L[1, i]
    end
    return y
end

# Backward solve: Lᵀ x = y   (writes x in place; safe to alias x === y)
function _band_backward_solve3!(x::AbstractVector{Float64},
                                 L::Matrix{Float64},
                                 n::Int,
                                 y::AbstractVector{Float64})
    @inbounds for i in n:-1:1
        s = y[i]
        khi = min(n, i + 3)
        for k in (i + 1):khi
            s -= L[1 + k - i, i] * x[k]
        end
        x[i] = s / L[1, i]
    end
    return x
end

# ── ldiv! dispatch ──────────────────────────────────────────────────────────

"""
    LinearAlgebra.ldiv!(out, F::BandedCholesky3, b) -> out

Solve `A x = b` where `A = L Lᵀ` is the factorization stored in `F`. Writes
the result into `out`. Allocation-free; `out` and `b` may not alias.
"""
function LinearAlgebra.ldiv!(out::AbstractVector{Float64},
                              F::BandedCholesky3,
                              b::AbstractVector{Float64})
    _band_forward_solve3!(out, F.L, F.n, b)
    _band_backward_solve3!(out, F.L, F.n, out)
    return out
end

"""
    LinearAlgebra.ldiv!(out, F::DenseSplineFactor, b) -> out

Periodic-BC fallback solve via dense LAPACK Cholesky. Allocation-free.
"""
function LinearAlgebra.ldiv!(out::AbstractVector{Float64},
                              F::DenseSplineFactor,
                              b::AbstractVector{Float64})
    copyto!(out, b)
    ldiv!(F.F, out)
    return out
end

# ── Backward-compat `\` operator ────────────────────────────────────────────
# The lower-level SAtransform / SIIntcoefficients APIs that take a factor
# positionally use `factor \ b`. Provide an allocating fallback that delegates
# to the in-place ldiv! method so those callers still work.

"""
    F \\ b -> Vector{Float64}

Allocating solve `A x = b` for `F::AbstractSplineFactor`. Provided for the
lower-level `SAtransform(sp, gammaBC, pqFactor, b)` /
`SIIntcoefficients(sp, gammaBC, p1Factor, uMish)` APIs that pass a factor
positionally. Hot-path callers should use the in-place
`LinearAlgebra.ldiv!(out, F, b)` form instead.
"""
function Base.:\(F::AbstractSplineFactor, b::AbstractVector{Float64})
    out = similar(b)
    return ldiv!(out, F, b)
end
