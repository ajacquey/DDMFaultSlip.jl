module TestConstantFriction

using DDMFaultSlip
using StaticArrays
using Statistics
using SpecialFunctions
using Interpolations
using LinearAlgebra
using Test

include("injection_utils.jl")

# Fluid properties
Δp = 0.4
Δpᵢ = 0.1
α = 0.04
# Elastic properties
μ = 6.6667e+02
ν = 0.0
# Friction properties
f = 0.5
h = 1.0e-05
kₙ = μ / h
kₛ = μ / h

function σ₀(X)
    return ones(length(X))
end

function p_func_2D(X::Vector{SVector{2,T}}, time::T)::Vector{T} where {T<:Real}
    return Δp * erfc.([abs(X[idx][1]) for idx in eachindex(X)] / sqrt(α * time))
end

function p_func_3D(X::Vector{SVector{3,T}}, time::T)::Vector{T} where {T<:Real}
    return Δpᵢ * expint.(1, [norm(X[idx])^2 for idx in eachindex(X)] / (α * time))
end

function ϵ_analytical_2D(mesh::DDMesh1D{T}, time::T)::Vector{T} where {T<:Real}
    return -p_func_2D([mesh.elems[i].X for i in 1:length(mesh.elems)], time) / kₙ
end

function ϵ_analytical_3D(mesh::DDMesh2D{T}, time::T)::Vector{T} where {T<:Real}
    return -p_func_3D([mesh.elems[i].X for i in 1:length(mesh.elems)], time) / kₙ
end

function δ_analytical_2D(mesh::DDMesh1D{T}, Ts::T, time::T)::Vector{T} where {T<:Real}
    x = [mesh.elems[i].X[1] for i in 1:length(mesh.elems)]
    δ = zeros(length(x))

    x_a, δ_a, λ = injection_analytical_gs(Ts, 500)
    a = λ * sqrt(α * time)

    itp = linear_interpolation(x_a * a, δ_a * a * f * Δp / μ) # create interpolation function

    # Outside of rupture front = 0
    δ[abs.(x).>=a] .= 0.0
    # Inside, interpolate
    δ[abs.(x).<a] = itp(x[abs.(x).<a])
    return δ
end

function δ_analytical_3D(mesh::DDMesh2D{T}, Ts::T, time::T)::Vector{T} where {T<:Real}
    r = [norm(mesh.elems[i].X) for i in 1:length(mesh.elems)]
    δ = zeros(length(r))

    r_a, δ_a, λ = injection_analytical_3D(Ts, 500)
    a = λ * sqrt(α * time)
    if (Ts > 1.0)
        itp = linear_interpolation(r_a * a, δ_a * a * f * Δpᵢ / μ) # create interpolation function
    elseif (Ts < 0.1)
        itp = linear_interpolation(r_a * a, δ_a * sqrt(α * time) * f * Δpᵢ / μ) # create interpolation function
    end

    # Outside of rupture front = 0
    δ[abs.(r).>=a] .= 0.0
    # Inside, interpolate
    δ[abs.(r).<a] = itp(r[abs.(r).<a])
    return δ
end

function calculateError(u, u_sol)
    idx = u_sol .> 0.0

    return mean(abs.((u[idx] .- u_sol[idx]) ./ u_sol[idx]))
end

# Length of the domain
function L(Ts)
    x_a, δ_a, λ = injection_analytical_gs(Ts, 500)
    return 1.1 * λ * sqrt(α * 10.0)
end

@testset "Fluid-induced aseismic slip" begin
    @testset "2D: T = 0.1" begin
        # Initial shear stress
        Ts = 0.1
        function τ₀(X)
            return f * (σ₀(X) .- Δp * Ts)
        end

        # Create mesh
        start_point = SVector(-L(Ts), 0.0)
        end_point = SVector(L(Ts), 0.0)
        N = 101
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = CoupledDDProblem2D(mesh; μ=μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_2D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantFriction(f, kₙ, kₛ))

        # Time sequence
        time_seq = collect(range(0.0, stop=10.0, length=11))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=10.0)

        # Run problem
        run!(problem, time_stepper; log=false)

        # Analytical solutions
        δ_sol = δ_analytical_2D(mesh, Ts, time_seq[end])

        # Error
        err = calculateError(problem.δ.value, δ_sol)
        # Error less than 2%
        @test err < 0.02
    end
    @testset "2D: T = 0.5" begin
        # Initial shear stress
        Ts = 0.5
        function τ₀(X)
            return f * (σ₀(X) .- Δp * Ts)
        end

        # Create mesh
        start_point = SVector(-L(Ts), 0.0)
        end_point = SVector(L(Ts), 0.0)
        N = 101
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = CoupledDDProblem2D(mesh; μ=μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_2D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantFriction(f, kₙ, kₛ))

        # Time sequence
        time_seq = collect(range(0.0, stop=10.0, length=11))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=10.0)

        # Run problem
        run!(problem, time_stepper; log=false)

        # Analytical solutions
        δ_sol = δ_analytical_2D(mesh, Ts, time_seq[end])

        # Error
        err = calculateError(problem.δ.value, δ_sol)
        # Error less than 2%
        @test err < 0.02
    end
    @testset "2D: T = 0.9" begin
        # Initial shear stress
        Ts = 0.9
        function τ₀(X)
            return f * (σ₀(X) .- Δp * Ts)
        end

        # Create mesh
        start_point = SVector(-L(Ts), 0.0)
        end_point = SVector(L(Ts), 0.0)
        N = 101
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = CoupledDDProblem2D(mesh; μ=μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_2D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantFriction(f, kₙ, kₛ))

        # Time sequence
        time_seq = collect(range(0.0, stop=10.0, length=11))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=10.0)

        # Run problem
        run!(problem, time_stepper; log=false)

        # Analytical solution
        δ_sol = δ_analytical_2D(mesh, Ts, time_seq[end])

        # Error
        err = calculateError(problem.δ.value, δ_sol)
        # Error less than 2%
        @test err < 0.02
    end
    @testset "3D: T = 0.01" begin
        Ts = 0.01
        function τ₀_x(X)
            return f * (σ₀(X) .- Δpᵢ * Ts)
        end

        function τ₀_y(X)
            return zeros(length(X))
        end

        # Create mesh
        mesh = DDMesh2D(Float64, "mesh.msh")

        # Create problem
        problem = CoupledDDProblem3D(mesh; μ=μ, ν=ν)

        # ICs
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, SVector(τ₀_x, τ₀_y))

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_3D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantFriction(f, kₛ, kₙ))

        time_seq = collect(range(0.0, stop=0.35, length=11))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=0.35)

        # Run problem
        run!(problem, time_stepper, log=false)

        # Analytical solutions
        δ_sol = δ_analytical_3D(mesh, Ts, time_seq[end])

        # Error
        err = calculateError(problem.δ_x.value, δ_sol)
        # Error less than 5%
        @test err < 0.05
    end
    @testset "3D: T = 4.0" begin
        Ts = 4.0
        function τ₀_x(X)
            return f * (σ₀(X) .- Δpᵢ * Ts)
        end

        function τ₀_y(X)
            return zeros(length(X))
        end

        # Create mesh
        mesh = DDMesh2D(Float64, "mesh.msh")

        # Create problem
        problem = CoupledDDProblem3D(mesh; μ=μ, ν=ν)

        # ICs
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, SVector(τ₀_x, τ₀_y))

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_3D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantFriction(f, kₛ, kₙ))

        time_seq = collect(range(0.0, stop=1.0e+03, length=11))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=1.0e+03)

        # Run problem
        run!(problem, time_stepper, log=false)

        # Analytical solutions
        δ_sol = δ_analytical_3D(mesh, Ts, time_seq[end])

        # Error
        err = calculateError(problem.δ_x.value, δ_sol)
        # Error less than 4%
        @test err < 0.04
    end
end

end