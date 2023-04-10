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

function w_analytical_2D(mesh::DDMesh1D{T}, time::T)::Vector{T} where {T<:Real}
    return -p_func_2D([mesh.elems[i].X for i in 1:length(mesh.elems)], time) / kₙ
end

function w_analytical_3D(mesh::DDMesh2D{T}, time::T)::Vector{T} where {T<:Real}
    return -p_func_3D([mesh.elems[i].X for i in 1:length(mesh.elems)], time) / kₙ
end

function Δσ(X::SVector{N,T}, Δw::T) where {N,T<:Real}
    return kₙ * Δw
end

function dΔσ(X::SVector{N,T}, Δw::T) where {N,T<:Real}
    return kₙ
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
        problem = NormalDDProblem(mesh; μ=μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_2D))

        # Elastic opening
        addConstraint!(problem, FunctionConstraint(Δσ, dΔσ))

        # Time sequence
        time_seq = collect(range(0.0, stop=10.0, length=21))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=10.0)

        # Run problem
        run!(problem, time_stepper; log=false)

        # Analytical solutions
        w_sol = w_analytical_2D(mesh, time_seq[end])

        # Error less than 4%
        @test isapprox(problem.w.value, w_sol; atol=4.0e-02)
    end
    @testset "Fluid-induced opening - 3D" begin
        # Create mesh
        mesh = DDMesh2D(Float64, "mesh.msh")

        # Elastic properties
        μ = 1.0
        ν = 0.0

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ, ν=ν)

        # ICs
        addNormalStressIC!(problem, σ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_3D))

        # Elastic opening
        addConstraint!(problem, FunctionConstraint(Δσ, dΔσ))

        # Time sequence
        time_seq = collect(range(0.0, stop=1.0, length=11))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=1.0)

        # Run problem
        run!(problem, time_stepper; log=false)

        # Analytical solutions
        w_sol = w_analytical_3D(mesh, time_seq[end])

        # Error less than 4%
        @test isapprox(problem.w.value, w_sol; atol=4.0e-02)
    end
end

end