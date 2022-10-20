module TestFluidCoupling

using DDMFaultSlip
using StaticArrays
using Statistics
using SpecialFunctions
using Test

# Fluid properties
Δp = 0.4
α = 0.04
# Elastic properties
μ = 6.6667e+02
# Friction properties
τ₀ = 10.0
h = 1.0e-03
kₙ = μ / h
kₛ = μ / h

function σ₀(X)
    return 1.0
end

function p_func(X::SVector{2, T}, time::T)::T where {T<:Real}
    return Δp * erfc(abs(X[1]) / sqrt(α * time))
end

function DD_analytical(mesh::DDMesh1D{T}, time::T)::Vector{T} where {T<:Real}
    return -p_func.([mesh.elems[i].X for i in 1:length(mesh.elems)], time) / kₙ
end

@testset "Fluid coupling problems" begin
    @testset "Fluid-induced opening" begin
        # Create mesh
        start_point = SVector(-1.0, 0.0)
        end_point = SVector(1.0, 0.0)
        N = 101
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = CoupledDDProblem2D(mesh; μ = μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantYield(τ₀, kₙ, kₛ))

        # Time sequence
        time_seq = collect(range(0.5, stop = 10.0, length = 20))
        time_stepper = TimeSequence(time_seq; start_time = 0.0, end_time = 10.0)

        # Run problem
        run!(problem, time_stepper; log = false)

        # Analytical solutions
        DD_sol = DD_analytical(mesh, μ)
        # Error
        err = mean(abs.(problem.ϵ.value - DD_sol) ./ DD_sol) 
        # Error less than 2%
        @test err < 0.02
    end
end

end