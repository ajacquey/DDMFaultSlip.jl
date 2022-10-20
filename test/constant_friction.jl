module TestConstantFriction

using DDMFaultSlip
using StaticArrays
using Statistics
using SpecialFunctions
using Interpolations
using Test

include("injection_utils.jl")

# Fluid properties
Δp = 0.4
α = 0.04
# Elastic properties
μ = 6.6667e+02
# Friction properties
f = 0.5
h = 1.0e-03
kₙ = μ / h
kₛ = μ / h

function σ₀(X)
    return 1.0
end

function p_func(X::SVector{2, T}, time::T)::T where {T<:Real}
    return Δp * erfc(abs(X[1]) / sqrt(α * time))
end

function ϵ_analytical(mesh::DDMesh1D{T}, time::T)::Vector{T} where {T<:Real}
    return -p_func.([mesh.elems[i].X for i in 1:length(mesh.elems)], time) / kₙ
end

function calculateError(u, v, u_sol, v_sol)
    idx = v_sol .> 0.0

    return mean(vcat(abs.((u[idx] .- u_sol[idx]) ./ u_sol[idx]), abs.((v[idx] .- v_sol[idx]) ./ v_sol[idx])))
end

@testset "Fluid-induced aseismic slip" begin
    @testset "T = 0.1" begin
        # Initial shear stress
        Ts = 0.1
        function τ₀(X)
            return f * (σ₀(X) - Δp * Ts)
        end

        # Length of the domain
        function L(T)
            x_a, δ_a, λ = injection_analytical_gs(T, 500)
            return 1.5 * λ * sqrt(α * 20.0)
        end

        # Analytical solution for slip
        function δ_analytical(mesh::DDMesh1D{T}, time::T)::Vector{T} where {T<:Real}
            x = [mesh.elems[i].X[1] for i in 1:length(mesh.elems)]
            δ = zeros(length(x))

            x_a, δ_a, λ = injection_analytical_gs(Ts, 500)
            a = λ * sqrt(α * time)

            itp = linear_interpolation(x_a * a, δ_a * a * f * Δp / μ) # create interpolation function

            # Outside of rupture front = 0
            δ[abs.(x) .>= a] .= 0.0
            # Inside, interpolate
            δ[abs.(x) .< a] = itp(x[abs.(x) .< a])
            return δ
        end

        # Create mesh
        start_point = SVector(-L(Ts), 0.0)
        end_point = SVector(L(Ts), 0.0)
        N = 101
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = CoupledDDProblem2D(mesh; μ = μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantFriction(f, kₙ, kₛ))

        # Time sequence
        time_seq = collect(range(0.5, stop = 10.0, length = 20))
        time_stepper = TimeSequence(time_seq; start_time = 0.0, end_time = 20.0)

        # Run problem
        run!(problem, time_stepper; log = false)

        # Analytical solutions
        ϵ_sol = ϵ_analytical(mesh, time_seq[end])
        δ_sol = δ_analytical(mesh, time_seq[end])

        # Error
        err = calculateError(problem.ϵ.value, problem.δ.value, ϵ_sol, δ_sol)
        # Error less than 2%
        @test err < 0.02
    end
    @testset "T = 0.5" begin
        # Initial shear stress
        Ts = 0.5
        function τ₀(X)
            return f * (σ₀(X) - Δp * Ts)
        end

        # Length of the domain
        function L(T)
            x_a, δ_a, λ = injection_analytical_gs(T, 500)
            return 1.5 * λ * sqrt(α * 20.0)
        end

        # Analytical solution for slip
        function δ_analytical(mesh::DDMesh1D{T}, time::T)::Vector{T} where {T<:Real}
            x = [mesh.elems[i].X[1] for i in 1:length(mesh.elems)]
            δ = zeros(length(x))

            x_a, δ_a, λ = injection_analytical_gs(Ts, 500)
            a = λ * sqrt(α * time)

            itp = linear_interpolation(x_a * a, δ_a * a * f * Δp / μ) # create interpolation function

            # Outside of rupture front = 0
            δ[abs.(x) .>= a] .= 0.0
            # Inside, interpolate
            δ[abs.(x) .< a] = itp(x[abs.(x) .< a])
            return δ
        end

        # Create mesh
        start_point = SVector(-L(Ts), 0.0)
        end_point = SVector(L(Ts), 0.0)
        N = 101
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = CoupledDDProblem2D(mesh; μ = μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantFriction(f, kₙ, kₛ))

        # Time sequence
        time_seq = collect(range(0.5, stop = 10.0, length = 20))
        time_stepper = TimeSequence(time_seq; start_time = 0.0, end_time = 20.0)

        # Run problem
        run!(problem, time_stepper; log = false)

        # Analytical solutions
        ϵ_sol = ϵ_analytical(mesh, time_seq[end])
        δ_sol = δ_analytical(mesh, time_seq[end])

        # Error
        err = calculateError(problem.ϵ.value, problem.δ.value, ϵ_sol, δ_sol)
        # Error less than 2%
        @test err < 0.02
    end
    @testset "T = 0.9" begin
        # Initial shear stress
        Ts = 0.9
        function τ₀(X)
            return f * (σ₀(X) - Δp * Ts)
        end

        # Length of the domain
        function L(T)
            x_a, δ_a, λ = injection_analytical_gs(T, 500)
            return 1.5 * λ * sqrt(α * 20.0)
        end

        # Analytical solution for slip
        function δ_analytical(mesh::DDMesh1D{T}, time::T)::Vector{T} where {T<:Real}
            x = [mesh.elems[i].X[1] for i in 1:length(mesh.elems)]
            δ = zeros(length(x))

            x_a, δ_a, λ = injection_analytical_gs(Ts, 500)
            a = λ * sqrt(α * time)

            itp = linear_interpolation(x_a * a, δ_a * a * f * Δp / μ) # create interpolation function

            # Outside of rupture front = 0
            δ[abs.(x) .>= a] .= 0.0
            # Inside, interpolate
            δ[abs.(x) .< a] = itp(x[abs.(x) .< a])
            return δ
        end

        # Create mesh
        start_point = SVector(-L(Ts), 0.0)
        end_point = SVector(L(Ts), 0.0)
        N = 101
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = CoupledDDProblem2D(mesh; μ = μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantFriction(f, kₙ, kₛ))

        # Time sequence
        time_seq = collect(range(0.5, stop = 10.0, length = 20))
        time_stepper = TimeSequence(time_seq; start_time = 0.0, end_time = 20.0)

        # Run problem
        run!(problem, time_stepper; log = false)

        # Analytical solutions
        ϵ_sol = ϵ_analytical(mesh, time_seq[end])
        δ_sol = δ_analytical(mesh, time_seq[end])

        # Error
        err = calculateError(problem.ϵ.value, problem.δ.value, ϵ_sol, δ_sol)
        # Error less than 2%
        @test err < 0.02
    end
end

end