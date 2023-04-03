module TestFluidCoupling

using DDMFaultSlip
using StaticArrays
using SpecialFunctions
using LinearAlgebra
using Test

# Fluid properties
α = 0.04
Δp = 0.4
Δpᵢ = 0.1
# Elastic properties
μ = 6.6667e+02
# Friction properties
τₛ = 10.0
h = 1.0e-03
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

function DD_analytical_2D(mesh::DDMesh1D{T}, time::T)::Vector{T} where {T<:Real}
    return -p_func_2D([mesh.elems[i].X for i in 1:length(mesh.elems)], time) / kₙ
end

function DD_analytical_3D(mesh::DDMesh2D{T}, time::T)::Vector{T} where {T<:Real}
    return -p_func_3D([mesh.elems[i].X for i in 1:length(mesh.elems)], time) / kₙ
end

@testset "Fluid coupling problems" begin
    @testset "Fluid-induced opening - 2D" begin
        # Create mesh
        start_point = SVector(-1.0, 0.0)
        end_point = SVector(1.0, 0.0)
        N = 101
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = CoupledDDProblem(mesh; μ=μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_2D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantYield(τₛ, kₙ, kₛ))

        # Time sequence
        time_seq = collect(range(0.5, stop=10.0, length=20))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=10.0)

        # Run problem
        run!(problem, time_stepper; log=true)

        # Analytical solutions
        DD_sol = DD_analytical_2D(mesh, μ)

        # Error less than 4%
        @test isapprox(problem.ϵ.value, DD_sol; atol=4.0e-02)
    end
    @testset "Fluid-induced opening - 3D" begin
        # Create mesh
        mesh = DDMesh2D(Float64, "mesh.msh")

        # Elastic properties
        μ = 1.0
        ν = 0.0

        # Create problem
        problem = CoupledDDProblem(mesh; μ=μ, ν=ν)

        # ICs
        addNormalStressIC!(problem, σ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_3D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantYield(τₛ, kₙ, kₛ))

        # Time sequence
        time_seq = collect(range(0.5, stop=5.0, length=10))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=5.0)

        # Run problem
        run!(problem, time_stepper; log=true, pc=false, hmat_eta=6.0)

        # Analytical solutions
        DD_sol = DD_analytical_3D(mesh, μ)

        # Error less than 4%
        @test isapprox(problem.ϵ.value, DD_sol; atol=4.0e-02)
    end
end

end