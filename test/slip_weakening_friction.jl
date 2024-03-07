module TestSlipWeakeningFriction

using DDMFaultSlip
using StaticArrays
using Statistics
using SpecialFunctions
using LinearAlgebra
using Test

# Fluid properties
Δp = 0.75
α = 0.04
# Elastic properties
μ = 6.6667e+02
ν = 0.0
# Friction properties
fₚ = 0.5
fᵣ = 0.3
δᵣ = (fₚ - fᵣ) / μ # aw = 1
h = 1.0e-05
k = μ / h
η = 100.0
# Time
t₀ = 0.0 # start time
nₜ = 20 # number of time steps
w = (fₚ - fᵣ) / δᵣ
δw = fₚ / w
aw = μ / fₚ * δw

function σ₀(X)
  	return ones(length(X))
end

function p_func_2D(X::Vector{SVector{2,T}}, time::T)::Vector{T} where {T<:Real}
  	return Δp * erfc.([abs(X[idx][1]) for idx in eachindex(X)] / sqrt(α * time))
end

@testset "Fluid-induced slip - slip-weakening friction" begin
    @testset "2D: fᵣ / fₚ = 0.6, Δp / σ₀ = $(Δp)  τ₀ / τₚ = 0.3" begin
        # Initial shear stress
		shear_stress_ratio = 0.3
        # End time
        tₑ = (15.0)^2 / (α / 4.0)
        # Initial shear stress
        function τ₀(X)
            return fₚ * σ₀(X) * shear_stress_ratio
        end

        # Create mesh
        start_point = SVector(-8.0, 0.0)
        end_point = SVector(8.0, 0.0)
        N = 101
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = ShearDDProblem(mesh; μ=μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_2D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, SlipWeakeningFriction(fₚ, fᵣ, δᵣ, k; η = η))

        # Time stepper
        time_stepper = ConstantDT(t₀, tₑ, nₜ)

        # Run problem
        run!(problem, time_stepper; log=false, nl_abs_tol=1.0e-06, nl_max_it=20)

        # Check max slip value
        @test maximum(problem.δ.value) > 0.6 * δw
    end
    @testset "2D: fᵣ / fₚ = 0.6, Δp / σ₀ = $(Δp)  τ₀ / τₚ = 0.4" begin
        # Initial shear stress
		shear_stress_ratio = 0.4
        # End time
        tₑ = (7.5)^2 / (α / 4.0)
        # Initial shear stress
        function τ₀(X)
            return fₚ * σ₀(X) * shear_stress_ratio
        end

        # Create mesh
        start_point = SVector(-7.0, 0.0)
        end_point = SVector(7.0, 0.0)
        N = 101
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = ShearDDProblem(mesh; μ=μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_2D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, SlipWeakeningFriction(fₚ, fᵣ, δᵣ, k ; η = η))

        # Time stepper
        time_stepper = ConstantDT(t₀, tₑ, nₜ)

        # Run problem
        run!(problem, time_stepper; log=false, nl_abs_tol=1.0e-06, nl_max_it=20)

        # Check max slip value
        @test maximum(problem.δ.value) > 1.2 * δw
    end
    @testset "2D: fᵣ / fₚ = 0.6, Δp / σ₀ = $(Δp)  τ₀ / τₚ = 0.5" begin
        # Initial shear stress
		shear_stress_ratio = 0.5
        # End time
        tₑ = (3.0)^2 / (α / 4.0)
        # Initial shear stress
        function τ₀(X)
            return fₚ * σ₀(X) * shear_stress_ratio
        end

        # Create mesh
        start_point = SVector(-7.0, 0.0)
        end_point = SVector(7.0, 0.0)
        N = 101
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = ShearDDProblem(mesh; μ=μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_2D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, SlipWeakeningFriction(fₚ, fᵣ, δᵣ, k; η = η))

        # Time stepper
        time_stepper = ConstantDT(t₀, tₑ, nₜ)

        # Run problem
        run!(problem, time_stepper; log=false, nl_abs_tol=1.0e-06, nl_max_it=20)

        # Check max slip value
        @test maximum(problem.δ.value) > 1.0 * δw
    end
end

end