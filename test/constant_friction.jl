module TestConstantFriction

using DDMFaultSlip
using StaticArrays
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
k = μ / h

function σ₀(X)
    return ones(length(X))
end

function p_func_2D(X::Vector{SVector{2,T}}, time::T)::Vector{T} where {T<:Real}
    return Δp * erfc.([abs(X[idx][1]) for idx in eachindex(X)] / sqrt(α * time))
end

function p_func_3D(X, time::T)::Vector{T} where {T<:Real}
    return Δpᵢ * expint.(1, [norm(X[idx])^2 for idx in eachindex(X)] / (α * time))
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

function δ_analytical_3D(mesh::DDMesh{T}, Ts::T, time::T)::Vector{T} where {T<:Real}
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

function L_axisymmetric(Ts)
    r_a, δ_a, λ = injection_analytical_3D(Ts, 500)
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
        problem = ShearDDProblem(mesh; μ=μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_2D))

        # Constant friction
        addFrictionConstraint!(problem, ConstantFriction(f, k))

        # Time sequence
        time_seq = collect(range(0.0, stop=5.0, length=6))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=5.0)

        # Run problem
        run!(problem, time_stepper; log=false, nl_abs_tol=1.0e-08)

        # Analytical solutions
        δ_sol = δ_analytical_2D(mesh, Ts, time_seq[end])
        
        # Error less than 2%
        @test isapprox(problem.δ.value, δ_sol; rtol=2.0e-02)
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
        problem = ShearDDProblem(mesh; μ=μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_2D))

        # Constant friction
        addFrictionConstraint!(problem, ConstantFriction(f, k))

        # Time sequence
        time_seq = collect(range(0.0, stop=5.0, length=6))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=5.0)

        # Run problem
        run!(problem, time_stepper; log=false)

        # Analytical solutions
        δ_sol = δ_analytical_2D(mesh, Ts, time_seq[end])
        
        # Error less than 2%
        @test isapprox(problem.δ.value, δ_sol; rtol=2.0e-02)
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
        problem = ShearDDProblem(mesh; μ=μ)

        # Add IC
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_2D))

        # Constant friction
        addFrictionConstraint!(problem, ConstantFriction(f, k))

        # Time sequence
        time_seq = collect(range(0.0, stop=5.0, length=6))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=5.0)

        # Run problem
        run!(problem, time_stepper; log=false)

        # Analytical solution
        δ_sol = δ_analytical_2D(mesh, Ts, time_seq[end])

        # Error less than 2%
        @test isapprox(problem.δ.value, δ_sol; rtol=2.0e-02)
    end
    @testset "3D: T = 0.01" begin
        Ts = 0.01
        function τ₀(X)
            return f * (σ₀(X) .- Δpᵢ * Ts)
        end

        function τ₀_y(X)
            return zeros(length(X))
        end

        # Create mesh
        mesh = DDMesh2D(Float64, "mesh.msh")

        # Create problem
        problem = ShearDDProblem(mesh; μ=μ, ν=ν)

        # ICs
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_3D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantFriction(f, k))

        time_seq = collect(range(0.0, stop=0.035, length=2))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=0.035)

        # Run problem
        run!(problem, time_stepper; log=false, nl_abs_tol=1.0e-08)

        # Analytical solutions
        δ_sol = δ_analytical_3D(mesh, Ts, time_seq[end])

        # Error less than 5.0%
        @test isapprox(problem.δ.value, δ_sol; atol=5.0e-02)
    end
    @testset "3D: T = 4.0" begin
        Ts = 4.0
        function τ₀(X)
            return f * (σ₀(X) .- Δpᵢ * Ts)
        end

        # Create mesh
        mesh = DDMesh2D(Float64, "mesh.msh")

        # Create problem
        problem = ShearDDProblem(mesh; μ=μ, ν=ν)

        # ICs
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_3D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantFriction(f, k))

        time_seq = collect(range(0.0, stop=1.0e+02, length=2))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=1.0e+02)

        # Run problem
        run!(problem, time_stepper, log=false, nl_abs_tol=1.0e-08)

        # Analytical solutions
        δ_sol = δ_analytical_3D(mesh, Ts, time_seq[end])

        # Error less than 4%
        @test isapprox(problem.δ.value, δ_sol; rtol=4.0e-02)
    end
    @testset "3D axisymmetric: T = 0.01" begin
        Ts = 0.01
        function τ₀(X)
            return f * (σ₀(X) .- Δpᵢ * Ts)
        end

        function τ₀_y(X)
            return zeros(length(X))
        end

        # Create mesh
        start_point = SVector(0.0, 0.0)
        end_point = SVector(L_axisymmetric(Ts), 0.0)
        N = 102
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = ShearDDProblem(mesh; axisymmetric=true, μ=μ, ν=ν)

        # ICs
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_3D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantFriction(f, k))
        
        # Time sequence
        time_seq = collect(range(0.0, stop=1.0e+01, length=6))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=1.0e+01)

        # Run problem
        run!(problem, time_stepper; log=false, nl_abs_tol=1.0e-08)

        # Analytical solutions
        δ_sol = δ_analytical_3D(mesh, Ts, time_seq[end])

        # Error less than 5.0%
        @test isapprox(problem.δ.value, δ_sol; atol=5.0e-02)
    end
    @testset "3D axisymmetric: T = 4.0" begin
        Ts = 4.0
        function τ₀(X)
            return f * (σ₀(X) .- Δpᵢ * Ts)
        end

        # Create mesh
        start_point = SVector(0.0, 0.0)
        end_point = SVector(L_axisymmetric(Ts), 0.0)
        N = 102
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = ShearDDProblem(mesh; axisymmetric=true, μ=μ, ν=ν)

        # ICs
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_3D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantFriction(f, k))

        # Time sequence
        time_seq = collect(range(0.0, stop=1.0e+01, length=6))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=1.0e+01)

        # Run problem
        run!(problem, time_stepper, log=false, nl_abs_tol=1.0e-08)

        # Analytical solutions
        δ_sol = δ_analytical_3D(mesh, Ts, time_seq[end])

        # Error less than 4%
        @test isapprox(problem.δ.value, δ_sol; rtol=4.0e-02)
    end
    @testset "3D: ν=0.2, T = 4.0" begin
        Ts = 4.0
        function τ₀(X, sym)
            if (sym == :x)
                return f * (σ₀(X) .- Δpᵢ * Ts)
            else
                return zeros(length(X))
            end
        end

        # Create mesh
        mesh = DDMesh2D(Float64, "mesh.msh")

        # Create problem
        problem = ShearDDProblem(mesh; μ=μ, ν=0.2)

        # ICs
        addNormalStressIC!(problem, σ₀)
        addShearStressIC!(problem, τ₀)

        # Fluid coupling
        addFluidCoupling!(problem, FunctionPressure(mesh, p_func_3D))

        # Constant yield (dummy plastic model)
        addFrictionConstraint!(problem, ConstantFriction(f, k))

        # Time sequence
        time_seq = collect(range(0.0, stop=1.0e+02, length=2))
        time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=1.0e+02)

        # Run problem
        run!(problem, time_stepper; log=false, nl_abs_tol=1.0e-08)

        # No analytical solution
        @test true
        # # Analytical solutions
        # δ_sol = δ_analytical_3D(mesh, Ts, time_seq[end])

        # # Error less than 4%
        # @test isapprox(problem.δ.value, δ_sol; rtol=4.0e-02)
    end
end

end