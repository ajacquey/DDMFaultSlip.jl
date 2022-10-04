module TestInjection

using DDMFaultSlip
using StaticArrays
using Interpolations
using SpecialFunctions
using Statistics
using Test

include("injection_utils.jl")

@testset "Fluid injection" begin
    @testset "Scaled 2D - T = 0.1" begin
        T = 0.1

        # Imposed shear stress
        function τ_inj(X, time)
            λ = lambda_analytical_gs(T, 500)
            return erfc(λ * abs(X[1])) - T
        end

        # Analytical solution
        function δ_analytical(mesh::DDMesh1D{Float64})
            x = [mesh.elems[i].X[1] for i in 1:length(mesh.elems)]

            x_a, δ_a, λ = injection_analytical_gs(T, length(x))
            itp = linear_interpolation(x_a, δ_a) # create interpolation function
            return itp(x)
        end

        # Create mesh
        start_point = SVector(-1.0, 0.0)
        end_point = SVector(1.0, 0.0)
        N = 100
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = ShearDDProblem2D(mesh; μ = μ)
        addConstraint!(problem, FunctionConstraint(τ_inj))

        # Run problem
        run!(problem; log = false)

        # Analytical solution
        δ_sol = δ_analytical(mesh)
        # Error
        err = mean(abs.(problem.δ.value - δ_sol) ./ δ_sol)
        # Error less than 2%
        @test err < 0.02
    end
    @testset "Scaled 2D - T = 0.5" begin
        T = 0.5

        # Imposed shear stress
        function τ_inj(X, time)
            λ = lambda_analytical_gs(T, 500)
            return erfc(λ * abs(X[1])) - T
        end

        # Analytical solution
        function δ_analytical(mesh::DDMesh1D{Float64})
            x = [mesh.elems[i].X[1] for i in 1:length(mesh.elems)]

            x_a, δ_a, λ = injection_analytical_gs(T, length(x))
            itp = linear_interpolation(x_a, δ_a) # create interpolation function
            return itp(x)
        end

        # Create mesh
        start_point = SVector(-1.0, 0.0)
        end_point = SVector(1.0, 0.0)
        N = 100
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = ShearDDProblem2D(mesh; μ = μ)
        addConstraint!(problem, FunctionConstraint(τ_inj))

        # Run problem
        run!(problem; log = false)

        # Analytical solution
        δ_sol = δ_analytical(mesh)
        # Error
        err = mean(abs.(problem.δ.value - δ_sol) ./ δ_sol)
        # Error less than 2%
        @test err < 0.02
    end
    @testset "Scaled 2D - T = 0.9" begin
        T = 0.9

        # Imposed shear stress
        function τ_inj(X, time)
            λ = lambda_analytical_gs(T, 500)
            return erfc(λ * abs(X[1])) - T
        end

        # Analytical solution
        function δ_analytical(mesh::DDMesh1D{Float64})
            x = [mesh.elems[i].X[1] for i in 1:length(mesh.elems)]

            x_a, δ_a, λ = injection_analytical_gs(T, length(x))
            itp = linear_interpolation(x_a, δ_a) # create interpolation function
            return itp(x)
        end

        # Create mesh
        start_point = SVector(-1.0, 0.0)
        end_point = SVector(1.0, 0.0)
        N = 100
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = ShearDDProblem2D(mesh; μ = μ)
        addConstraint!(problem, FunctionConstraint(τ_inj))

        # Run problem
        run!(problem; log = false)

        # Analytical solution
        δ_sol = δ_analytical(mesh)
        # Error
        err = mean(abs.(problem.δ.value - δ_sol) ./ δ_sol)
        # Error less than 2%
        @test err < 0.02
    end
end

end