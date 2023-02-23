module LinearSolvers

using DDMFaultSlip
using StaticArrays
using Test

function σ_cst(X, time::T) where {T<:Real}
  return -1.0
end

function ϵ_analytical_2D(mesh::DDMesh1D{T}, μ::T)::Vector{T} where {T<:Real}
    x = [mesh.elems[i].X[1] for i in 1:length(mesh.elems)]

    return sqrt.(1.0 .- x .^ 2) / μ
end

@testset "Linear solvers" begin
    @testset "Restarted GMRES" begin
        # Create mesh
        start_point = SVector(-1.0, 0.0)
        end_point = SVector(1.0, 0.0)
        N = 100
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ)
        addConstraint!(problem, FunctionConstraint(σ_cst))

        # Run problem
        run!(problem; log=false, l_solver="gmres")

        # Analytical solution
        ϵ_sol = ϵ_analytical_2D(mesh, μ)

        # Error less than 2%
        @test isapprox(problem.ϵ.value, ϵ_sol; rtol=2.0e-02)
    end
    @testset "IDR(s)" begin
        # Create mesh
        start_point = SVector(-1.0, 0.0)
        end_point = SVector(1.0, 0.0)
        N = 100
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ)
        addConstraint!(problem, FunctionConstraint(σ_cst))

        # Run problem
        run!(problem; log=false, l_solver="idrs")

        # Analytical solution
        ϵ_sol = ϵ_analytical_2D(mesh, μ)

        # Error less than 2%
        @test isapprox(problem.ϵ.value, ϵ_sol; rtol=2.0e-02)
    end
    @testset "BiCGStab(l)" begin
        # Create mesh
        start_point = SVector(-1.0, 0.0)
        end_point = SVector(1.0, 0.0)
        N = 100
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ)
        addConstraint!(problem, FunctionConstraint(σ_cst))

        # Run problem
        run!(problem; log=false, l_solver="bicgstabl")

        # Analytical solution
        ϵ_sol = ϵ_analytical_2D(mesh, μ)

        # Error less than 2%
        @test isapprox(problem.ϵ.value, ϵ_sol; rtol=2.0e-02)
    end
end
end