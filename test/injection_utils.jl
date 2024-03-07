using FastGaussQuadrature
using SpecialFunctions
using LinearAlgebra
using HypergeometricFunctions
using Roots

function stress_parameter_3D(λ::Float64)
    return 2.0 - Base.MathConstants.γ + 2.0 * λ^2 / 3.0 * pFq([1.0, 1.0], [2.0, 2.5], -λ^2) - log(4.0 * λ^2)
end

function amplification_factor_3D(T::Float64)
    return abs(fzero(λ->stress_parameter_3D(λ) - T, 1.0))
end

function low_T_slip_profile_3D(r::Float64, T::Float64)::Float64
    return 2.0 * sqrt(2.0 * T) / π * (acos(abs(r)) / abs(r) - sqrt(1.0 - r^2))
end

function high_T_slip_profile_3D(r::Float64)::Float64
    return 8.0 / π * (sqrt(1.0 - r^2) - abs(r) * acos(abs(r)))
end

function injection_analytical_3D(T::Float64, N::Int64=100)
    r = collect(range(0.0, stop=1.0, length=N))[2:end]

    λ = amplification_factor_3D(T)

    if (T > 2.0)
        δ = high_T_slip_profile_3D.(r)
    elseif (T < 0.1)
        δ = low_T_slip_profile_3D.(r, T)
    else
        throw(ErrorException("Solution not implemented yet!"))
    end

    return (r, δ, λ)
end

"""
    injection_analytical_gs(T::Float64, N::Int64 = 100)

Solves the slip distribution using Gauss-Chebyshev quadratures due to an
analytical pressure expression.
Inputs:
    T: the stress parameter
    N: number of quadrature points
Outputs:
    x: the Gauss-Checbyshev quadrature points
    δ: the slip distribution
    λ: the slip to fluid migration factor
"""
function injection_analytical_gs(T::Float64, N::Int64=100)
    # Solve for λ
    λ = lambda_analytical_gs(T, N)

    # Slip
    (x, δ) = slip_distribution_gs(λ, N)

    return (x, δ, λ)
end

"""
    lambda_analytical_gs(
        T::Float64,
        N::Int64,
        max_iters::Int64 = 200,
        abs_tol::Float64 = 1.0e-10,
        debug::Bool = False,
    )

Use Newton-Raphson iterations to solve for λ using Gauss-Chebyshev quadratures.
Inputs:
    T: the stress parameter
    N: number of quadrature points
    max_iters: the maximum number of Newton-Raphson iterations
    abs_tol: the absolute tolerance for the Newton solve
    debug: print convergence information
Outputs:
    λ: the slip to fluid migration factor
"""
function lambda_analytical_gs(
    T::Float64,
    N::Int64,
    max_iters::Int64=200,
    abs_tol::Float64=1.0e-10,
    debug::Bool=false,
)
    # Initialization
    λ = 1.0
    k = 0
    # Create Gauss-Chebyshev quadrature points
    s, w = gausschebyshevt(N)

    while (k < max_iters)
        # Update residuals and jacobian
        Res = residual_analytical(T, s, w, λ)
        Jac = jacobian_analytical(T, s, w, λ)

        # Check convergence
        if (abs(Res) <= abs_tol)
            if (debug)
                println("Solve converged! λ = ", λ, " after ", k, " iterations.")
            end
            return λ
        end

        # Update λ
        dλ = -Res / Jac
        λ += dλ
        k += 1
    end

    throw(ErrorException("Maximm number of iterations reached in the Newton solve!!!\n"))
end

"""
    residual_analytical(T::Float64, s::Vector{Float64}, w::Vector{Float64}, λ::Float64)

The residual for calculating λ
"""
function residual_analytical(T::Float64, s::Vector{Float64}, w::Vector{Float64}, λ::Float64)
    f(s) = erfc(λ * abs(s))
    return dot(w, f.(s)) / pi - T
end

"""
    jacobian_analytical(T::Float64, s::Vector{Float64}, w::Vector{Float64}, λ::Float64)

The jacobian for calculating λ
"""
function jacobian_analytical(T::Float64, s::Vector{Float64}, w::Vector{Float64}, λ::Float64)
    f(s) = -2 * abs(s) / sqrt(pi) * exp(-λ^2 * abs(s)^2)
    return dot(w, f.(s)) / pi
end

"""
    dF(s::Float64, u::Float64, λ::Float64)

The function dF from Viesca and Garagash (2018)
"""
function dF(s::Float64, u::Float64, λ::Float64)
    return 1 / π * erfc(λ * abs(s)) / (u - s)
end

"""
    F(u::Float64, λ::Float64, N::Int64)

The function F from Viesca and Garagash (2018)
"""
function F(u::Float64, λ::Float64, N::Int64)
    # Gauss-Chebyshev quadrature points
    s, w = gausschebyshevt(N)
    return dot(w, dF.(s, u, λ))
end

"""
    slip_gs!(δ::Vector{Float64}, x::Vector{Float64}, N::Int64, λ::Float64)

Slip evaluated with Gauss-Chebyshev quadrature
"""
function slip_gs!(δ::Vector{Float64}, x::Vector{Float64}, N::Int64, λ::Float64)
    # Gauss-Chebyshev quadrature points
    (s, w) = gausschebyshevu(N - 1)

    # Slip weigth from Viesca and Garagash (2018)
    Φ = [0.5 * (sin(k * acos(xi)) / k - sin((k + 2) * acos(xi)) / (k + 2)) for xi in x, k = N:-1:1]
    B = [2 * sin(acos(sj)) * sin((k + 1) * acos(sj)) / (N + 1) for k = N:-1:1, sj in s]
    S = zeros(N, N - 1)
    mul!(S, Φ, B)
    mul!(δ, S, F.(s, λ, N))
end

"""
    theta(x::Float64)

Theta function from Viesca and Garagash (2018)
"""
function theta(x::Float64)
    return acos(x::Float64)
end

"""
    slip_distribution_gs(λ::Float64, N::Int64)

Computes the slip distribution based on Gauss-Chebyshev quadratures
Inputs:
    λ: the slip to fluid migration factor
    N: number of quadrature points
Outputs
    x: the spatial discretization
    δ: the slip distribution
    
"""
function slip_distribution_gs(λ::Float64, N::Int64)
    x = collect(range(-1.0, 1.0, length=N))

    δ = similar(x)
    slip_gs!(δ, x, N, λ)
    return (x, δ)
end
