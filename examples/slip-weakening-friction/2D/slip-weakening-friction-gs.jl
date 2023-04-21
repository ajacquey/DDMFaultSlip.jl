using GaussChebyshevFracture
using NLsolve
using LinearAlgebra
using SpecialFunctions
using Parameters
using Setfield
using DelimitedFiles

# ###### FRICTION ######
# function friction(x::Vector{Float64}, par)::Vector{Float64}
#     @unpack t, A, fᵣ_fₚ, Δp_σ₀, τ₀_τₚ, n, gc  = par
#     δ = GaussChebyshevFracture.u(gc, x)
#     f_fₚ = similar(δ)
#     fill!(f_fₚ, fᵣ_fₚ)
#     f_fₚ[δ .<= 1.0 - fᵣ_fₚ] .= 1 .- δ[δ .<= 1.0 - fᵣ_fₚ]
#     return f_fₚ
# end

# ###### FRICTION DERIVATIVE ######
# function friction_derivative(x::Vector{Float64}, par)::Matrix{Float64}
#     @unpack t, A, fᵣ_fₚ, Δp_σ₀, τ₀_τₚ, n, gc  = par
#     δ = GaussChebyshevFracture.u(gc, x)
#     df_fₚ = zeros(size(gc.S))
#     df_fₚ[δ .<= 1.0 - fᵣ_fₚ, :] .= -gc.S[δ .<= 1.0 - fᵣ_fₚ, :]
#     return df_fₚ
# end

# ###### RESIDUAL ######
# function Residual!(x, par)
#     @unpack t, A, fᵣ_fₚ, Δp_σ₀, τ₀_τₚ, n, gc  = par
#     f_fₚ = friction(x[1:n], par)
#     if t == 0.0
#         R .= (f_fₚ .- τ₀_τₚ) / Δp_σ₀ .- (A * x[1:n]) / (abs(x[n+1]) * Δp_σ₀)
#     else
#         R .= (f_fₚ .- τ₀_τₚ) / Δp_σ₀ .- f_fₚ .* erfc.(abs.(x[n+1] * gc.x / t)) .- (A * x[1:n]) / (abs(x[n+1]) * Δp_σ₀)
#     end
# end

# ###### JACOBIAN ######
# function Jacobian!(x, par)
#     @unpack t, A, fᵣ_fₚ, Δp_σ₀, τ₀_τₚ, n, gc  = par
#     f_fₚ = friction(x[1:n], par)
#     df_fₚ = friction_derivative(x[1:n], par)

#     # J = zeros(n+1, n+1)
#     if t == 0.0
#         J[1:n+1, 1:n] .= df_fₚ / Δp_σ₀ .- A / (abs(x[n+1]) * Δp_σ₀)
#         J[:, n+1] .= (A * x[1:n]) / (abs(x[n+1])^2 * Δp_σ₀)
#     else
#         J[1:n+1, 1:n] .= df_fₚ .* (1.0 / Δp_σ₀ .- erfc.(abs.(x[n+1] * gc.x / t))) .- A / (abs(x[n+1]) * Δp_σ₀)
#         J[:, n+1] .= 2 * f_fₚ .* abs.(gc.x / t) .* exp.(-(gc.x * abs(x[n+1])).^2 / t^2) / sqrt(π) .+ (A * x[1:n]) / (abs(x[n+1])^2 * Δp_σ₀) 
#     end
#     # return J
# end

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

    # # Initial conditions
    # u0 = zeros(n+1)
    # u0[n+1] = 0.1

    # Gauss-Chebyshec quadrature
    gc = GaussChebyshev(n, 2)

    # Matrix
    A = [gc.w[j] / (π * (gc.s[j] - gc.x[i])) for i in eachindex(gc.x), j in eachindex(gc.s)]

    # Parameters
    # par = (t = 0.0, A = A, fᵣ_fₚ = fᵣ_fₚ, Δp_σ₀ = Δp_σ₀, τ₀_τₚ = τ₀_τₚ, n = n, gc = gc)

    # Initial time, end time and time steps
    t₀ = 0.0
    tₑ = 15.0
    nₜ = 200
    time = collect(range(start=t₀, stop=tₑ, length=nₜ))
    time = time[2:end]
    t = time[1]

    # Output file
    header = "t,slip,crack_length\n"
    filename = string(@__DIR__, "/slip-weakening-friction-gs.csv")
    open(filename; write=true) do f
        write(f, header) # header
    end


    ###### FRICTION ######
    function friction(x::Vector{Float64})::Vector{Float64}
        # @unpack t, A, fᵣ_fₚ, Δp_σ₀, τ₀_τₚ, n, gc  = par
        δ = GaussChebyshevFracture.u(gc, x)
        f_fₚ = similar(δ)
        fill!(f_fₚ, fᵣ_fₚ)
        f_fₚ[δ .<= 1.0 - fᵣ_fₚ] .= 1 .- δ[δ .<= 1.0 - fᵣ_fₚ]
        return f_fₚ
    end

    ###### FRICTION DERIVATIVE ######
    function friction_derivative(x::Vector{Float64})::Matrix{Float64}
        # @unpack t, A, fᵣ_fₚ, Δp_σ₀, τ₀_τₚ, n, gc  = par
        δ = GaussChebyshevFracture.u(gc, x)
        df_fₚ = zeros(size(gc.S))
        df_fₚ[δ .<= 1.0 - fᵣ_fₚ, :] .= -gc.S[δ .<= 1.0 - fᵣ_fₚ, :]
        return df_fₚ
    end

    # Solution
    u = zeros(n+1)
    u[n+1] = 0.001
    a = 0.0
    δ = 0.0

    for i in eachindex(time)
        # Time
        # @set par.t = time[i]
        t = time[i]
        println("Time: ", t)

        ###### RESIDUAL ######
        function Residual!(R, x)
            # @unpack t, A, fᵣ_fₚ, Δp_σ₀, τ₀_τₚ, n, gc  = par
            f_fₚ = friction(x[1:n])
            if t == 0.0
                R[1:n+1] .= (f_fₚ .- τ₀_τₚ) / Δp_σ₀ .- (A * x[1:n]) / (abs(x[n+1]) * Δp_σ₀)
            else
                R[1:n+1] .= (f_fₚ .- τ₀_τₚ) / Δp_σ₀ .- f_fₚ .* erfc.(abs.(x[n+1] * gc.x / t)) .- (A * x[1:n]) / (abs(x[n+1]) * Δp_σ₀)
            end
        end

        ###### JACOBIAN ######
        function Jacobian!(J, x)
            # @unpack t, A, fᵣ_fₚ, Δp_σ₀, τ₀_τₚ, n, gc  = par
            f_fₚ = friction(x[1:n])
            df_fₚ = friction_derivative(x[1:n])

            # J = zeros(n+1, n+1)
            if t == 0.0
                J[1:n+1, 1:n] .= df_fₚ / Δp_σ₀ .- A / (abs(x[n+1]) * Δp_σ₀)
                J[:, n+1] .= (A * x[1:n]) / (abs(x[n+1])^2 * Δp_σ₀)
            else
                J[1:n+1, 1:n] .= df_fₚ .* (1.0 / Δp_σ₀ .- erfc.(abs.(x[n+1] * gc.x / t))) .- A / (abs(x[n+1]) * Δp_σ₀)
                J[:, n+1] .= 2 * f_fₚ .* abs.(gc.x / t) .* exp.(-(gc.x * abs(x[n+1])).^2 / t^2) / sqrt(π) .+ (A * x[1:n]) / (abs(x[n+1])^2 * Δp_σ₀) 
            end
            # return J
        end

        # Solve
        res = nlsolve(Residual!, Jacobian!, vcat(u[1:n], u[end]+0.01), method = :newton, show_trace=true)

        # Update solution
        u = res.zero
        a = u[end]
        δ = maximum(GaussChebyshevFracture.u(gc, u[1:n]))

        println("\tCrack length: ", a)
        println("\tMax slip: ", δ)

        # Write results in file
        open(string(@__DIR__, "/slip-weakening-friction-gs.csv"); append=true) do f
            writedlm(f, hcat(t, δ, a), ',')
        end

    end
end


#     # Define the sup norm
#     norminf(x) = norm(x, Inf)

#     # Eigensolver
#     eigls = EigKrylovKit(dim = 70)

#     # Options for Newton solver, we pass the eigensolverr
#     opt_newton = BK.NewtonPar(tol = 1e-8, verbose = true, maxIter = 20, eigsolver = eigls)

#     # Options for continuation
#     opts_br = ContinuationPar(pMax = 15.0, pMin = 0.2,
#         # for a good looking curve
#         dsmin = 0.001, dsmax = 0.02, ds = 0.005,
#         # θ = 0.5,
#         # number of eigenvalues to compute
#         nev = 5,
#         # nev = 48,
#         # plotEveryStep = 10,# newtonOptions = (@set opt_newton.verbose = true),
#         maxSteps = 2000,
#         # detect codim 1 bifurcations
#         detectBifurcation = 3,
#         # Optional: bisection options for locating bifurcations
#         # nInversion = 4, dsminBisection = 1e-7, maxBisectionSteps = 25,
#         newtonOptions = opt_newton,
#         )
    
#     # Optional arguments for continuation
#     kwargsC = (plot = false, normC = norm,
#         verbosity = 2,
#         recordFromSolution = (x, t) -> (δ = norminf(GaussChebyshevFracture.u(gc, x[1:n])), a = x[n + 1]),
#         finaliseSolution = (z, tau, step, br; k...) -> finaliseSolution(z, tau, step, br; k...)
#     )

#     # Output file
#     header = "t,slip,crack_length,stability,bp,hp\n"
#     filename = string(@__DIR__, "/slip-weakening-friction-gs.csv")
#     open(filename; write=true) do f
#         write(f, header) # header
#     end

#     # Finalise solution ######
#     function finaliseSolution(z, tau, step, br; kwargs...)
#         x = z.u
#         t = z.p
#         # Get variables
#         δ = GaussChebyshevFracture.u(gc, x[1:n])
#         a = x[n+1]
#         # Infinite norm
#         δₙ = norminf(δ)
#         # Get stability and bf
#         b_point = 0
#         h_point = 0
#         if length(br.specialpoint)>0
#             if br.specialpoint[end].step == step
#                 if br.specialpoint[end].type == :bp
#                     b_point = 1
#                 end
#                 if br.specialpoint[end].type == :hp
#                     h_point = 1
#                 end
#             end
#         end

#         # Write results in file
#         open(string(@__DIR__, "/slip-weakening-friction-gs.csv"); append=true) do f
#             writedlm(f, hcat(t, δₙ, a, br.stable[step], b_point, h_point), ',')
#         end
#         return true # to continue
#     end

#     # Plot solution
#     function plotSolution!(x, t; k...)
#         plot!(t, norminf(GaussChebyshevFracture.u(gc, x[1:n])))
#     end

#     prob = BifurcationProblem(Residual, u0, par, (@lens _.t); J = Jacobian,
#         recordFromSolution = (x, t) -> (δ = norminf(GaussChebyshevFracture.u(gc, x[1:n])), a = x[n + 1]),
#         plotSolution = (x, t; k...) -> plotSolution!(x, t; k...),
#     )

#     br, x_final = BK.continuation(prob, Natural(), opts_br; kwargsC...)
# end

main()