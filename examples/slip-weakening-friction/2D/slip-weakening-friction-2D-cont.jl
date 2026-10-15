using GaussChebyshevFracture
import BifurcationKit as BK
import BifurcationKit: @optic, @set
using LinearAlgebra
using SpecialFunctions
using Parameters
using Setfield
using DelimitedFiles

###### FLUID PRESSURE ######
function fluid_pressure(x::Vector{Float64}, a::Float64, sqrt_t::Float64)
    if sqrt_t == 0.0
        return 0.0
    else
        return erfc.(abs.(a * x) / sqrt_t)
    end
end

###### FLUID PRESSURE DERIVATIVE ######
function dfluid_pressure(x::Vector{Float64}, a::Float64, sqrt_t::Float64)
    if sqrt_t == 0.0
        return 0.0
    else
        return -2 * abs.(x) ./ (sqrt(π) * sqrt_t) .* exp.(-(a * x / sqrt_t).^2) 
    end
end

###### FRICTION ######
function friction(x::Vector{Float64}, par)::Vector{Float64}
    @unpack sqrt_t, A, fᵣ_fₚ, Δp_σ₀, τ₀_τₚ, n, gc  = par
    δ = GaussChebyshevFracture.u(gc, x)
    f_fₚ = similar(δ)
    fill!(f_fₚ, fᵣ_fₚ)
    f_fₚ[δ .<= 1.0 - fᵣ_fₚ] .= 1 .- δ[δ .<= 1.0 - fᵣ_fₚ]
    return f_fₚ
end

###### FRICTION DERIVATIVE ######
function friction_derivative(x::Vector{Float64}, par)::Matrix{Float64}
    @unpack sqrt_t, A, fᵣ_fₚ, Δp_σ₀, τ₀_τₚ, n, gc  = par
    δ = GaussChebyshevFracture.u(gc, x)
    df_fₚ = zeros(size(gc.S))
    df_fₚ[δ .<= 1.0 - fᵣ_fₚ, :] .= -gc.S[δ .<= 1.0 - fᵣ_fₚ, :]
    return df_fₚ
end

###### RESIDUAL ######
function Residual(x, par)
    @unpack sqrt_t, A, fᵣ_fₚ, Δp_σ₀, τ₀_τₚ, n, gc  = par
    f_fₚ = friction(x[1:n], par)
    
    return (f_fₚ .- τ₀_τₚ) / Δp_σ₀ .- f_fₚ .* fluid_pressure(gc.x, x[n+1], sqrt_t) .- (A * x[1:n]) / (abs(x[n+1]) * Δp_σ₀)
end

###### JACOBIAN ######
function Jacobian(x, par)
    @unpack sqrt_t, A, fᵣ_fₚ, Δp_σ₀, τ₀_τₚ, n, gc  = par
    f_fₚ = friction(x[1:n], par)
    df_fₚ = friction_derivative(x[1:n], par)

    J = zeros(n+1, n+1)
    if sqrt_t == 0.0
        J[1:n+1, 1:n] .= df_fₚ / Δp_σ₀ .- A / (abs(x[n+1]) * Δp_σ₀)
        J[:, n+1] .= (A * x[1:n]) / (abs(x[n+1])^2 * Δp_σ₀)
    else
        J[1:n+1, 1:n] .= df_fₚ .* (1.0 / Δp_σ₀ .- fluid_pressure(gc.x, x[n+1], sqrt_t)) .- A / (abs(x[n+1]) * Δp_σ₀)
        J[:, n+1] .= - f_fₚ .* dfluid_pressure(gc.x, x[n+1], sqrt_t) .+ (A * x[1:n]) / (abs(x[n+1])^2 * Δp_σ₀) 
    end
    return J
end

# ###### PLOT SOLUTION ######
# function plotSol(x, t; k...)
#     plot!(gamma_dot(getindex(splitVar(x, 2), 1), getindex(splitVar(x, 2), 2), par), label=L"\dot{\gamma}^{p}" ; k...)
#     plot!(getindex(splitVar(x, 2), 1), label=L"\theta" ; k...)
#     plot!(getindex(splitVar(x, 2), 2), label=L"p" ; k...)
# end

function main()
    # Physical parameters ######
    fᵣ_fₚ = parse(Float64, ARGS[1])
    Δp_σ₀ = parse(Float64, ARGS[2])
    τ₀_τₚ = parse(Float64, ARGS[3])

    # Gauss-Chebyshev quadrature points
    n = 500

    # Initial conditions
    u0 = zeros(n+1)
    u0[n+1] = 0.1

    # Gauss-Chebyshec quadrature
    gc = GaussChebyshev(n, 2)

    # Matrix
    A = [gc.w[j] / (π * (gc.s[j] - gc.x[i])) for i in eachindex(gc.x), j in eachindex(gc.s)]

    # Parameters
    par = (sqrt_t = 0.2, A = A, fᵣ_fₚ = fᵣ_fₚ, Δp_σ₀ = Δp_σ₀, τ₀_τₚ = τ₀_τₚ, n = n, gc = gc)

    # Define the sup norm
    norminf(x) = norm(x, Inf)

    # Eigensolver
    # eigls = EigKrylovKit(dim = 70)

    # Options for Newton solver
    opt_newton = BK.NewtonPar(tol = 1e-8, verbose = true, max_iterations = 20)

    # Options for continuation
    opt_cont = BK.ContinuationPar(p_max = 15.0, p_min = 0.2,
        # for a good looking curve
        dsmin = 0.001, dsmax = 0.02, ds = 0.005,
        # θ = 0.5,
        # number of eigenvalues to compute
        # nev = 5,
        # nev = 48,
        # plotEveryStep = 10,# newtonOptions = (@set opt_newton.verbose = true),
        max_steps = 2000,
        # detect codim 1 bifurcations
        # detect_bifurcation = 3,
        # Optional: bisection options for locating bifurcations
        # nInversion = 4, dsminBisection = 1e-7, maxBisectionSteps = 25,
        newton_options = opt_newton,
        )
    
    # # Optional arguments for continuation
    # kwargsC = (plot = false, normC = norm,
    #     verbosity = 2,
    #     # recordFromSolution = (x, t) -> (δ = norminf(GaussChebyshevFracture.u(gc, x[1:n])), a = x[n + 1]),
    #     finalise_solution = (z, tau, step, br; k...) -> finaliseSolution(z, tau, step, br; k...)
    # )

    # Output file
    header = "sqrt_t,slip,crack_length,stability,bp,hp\n"
    filename = string(@__DIR__, "/outputs/slip_weakening_friction_2D_cont_f_$(fᵣ_fₚ)_p_$(Δp_σ₀)_tau_$(τ₀_τₚ).csv")
    open(filename; write=true) do f
        write(f, header) # header
    end

    # Finalise solution ######
    function finaliseSolution(z, tau, step, br; kwargs...)
        x = z.u
        sqrt_t = z.p
        # Get variables
        δ = GaussChebyshevFracture.u(gc, x[1:n])
        a = x[n+1]
        # Infinite norm
        δₙ = norminf(δ)
        # Get stability and bf
        b_point = 0
        h_point = 0
        if length(br.specialpoint)>0
            if br.specialpoint[end].step == step
                if br.specialpoint[end].type == :bp
                    b_point = 1
                end
                if br.specialpoint[end].type == :hp
                    h_point = 1
                end
            end
        end

        # Write results in file
        open(string(@__DIR__, "/outputs/slip_weakening_friction_2D_cont_f_$(fᵣ_fₚ)_p_$(Δp_σ₀)_tau_$(τ₀_τₚ).csv"); append=true) do f
            writedlm(f, hcat(sqrt_t, δₙ, a, br.stable[step], b_point, h_point), ',')
        end
        return true # to continue
    end

    # Plot solution
    function plotSolution!(x, sqrt_t; k...)
        plot!(sqrt_t, norminf(GaussChebyshevFracture.u(gc, x[1:n])))
    end

    prob = BK.BifurcationProblem(Residual, u0, par, (@optic _.sqrt_t); J = Jacobian,
        # recordFromSolution = (x, t) -> (δ = norminf(GaussChebyshevFracture.u(gc, x[1:n])), a = x[n + 1]),
        # plotSolution = (x, t; k...) -> plotSolution!(x, t; k...),
    )

    br = BK.continuation(prob, BK.PALC(), opt_cont; finalise_solution = (z, tau, step, br; k...) -> finaliseSolution(z, tau, step, br; k...),)
    nothing
end

main()