module TestDugdaleYield

using DDMFaultSlip
using StaticArrays
using Statistics
using SpecialFunctions
using Interpolations
using LinearAlgebra
using Test

# Elastic properties
μ = 1.0
ν = 0.25
h = 1.0e-05
k = μ / h

function get_crack_ratio(elems, σ, r)
    idx = findall(x -> x>= 0.99 * (1.0 - r), σ)
    if length(idx) > 0
        return elems[idx[end]].X[1]
    else
        return NaN
    end
end

@testset "Dugdale cohezive zone" begin
    @testset "2D: r = 0.25" begin
        # Initial shear stress
        r = 0.25
        # Dugdale crack ratio
        m = 1 / cos(π / 2 * r)

        # Create mesh
        start_point = SVector(-1.2*m, 0.0)
        end_point = SVector(1.2*m, 0.0)
        N = 201
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ)

        # Dugdale yield
        addCohesiveConstraint!(problem, DugdaleCohesiveZone(1.0 - r, -r, 1.0, k))

        outputs = [CSVDomainOutput(mesh, "outputs/test_dugdale_$(r)")]

        # Run problem
        run!(problem; outputs=outputs, log=false)

        # Analytical solution
        m_calc = get_crack_ratio(mesh.elems, problem.σ.value, r)

        # Error less than 3%
        @test isapprox(m_calc, m; rtol=3.0e-02)
    end
    @testset "2D: r = 0.5" begin
        # Initial shear stress
        r = 0.5
        # Dugdale crack ratio
        m = 1 / cos(π / 2 * r)

        # Create mesh
        start_point = SVector(-1.2*m, 0.0)
        end_point = SVector(1.2*m, 0.0)
        N = 201
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ)

        # Dugdale yield
        addCohesiveConstraint!(problem, DugdaleCohesiveZone(1.0 - r, -r, 1.0, k))

        outputs = [CSVDomainOutput(mesh, "outputs/test_dugdale_$(r)")]

        # Run problem
        run!(problem; outputs=outputs, log=false)

        # Analytical solution
        m_calc = get_crack_ratio(mesh.elems, problem.σ.value, r)

        # Error less than 3%
        @test isapprox(m_calc, m; rtol=3.0e-02)
    end
    @testset "2D: r = 0.75" begin
        # Initial shear stress
        r = 0.75
        # Dugdale crack ratio
        m = 1 / cos(π / 2 * r)

        # Create mesh
        start_point = SVector(-1.2*m, 0.0)
        end_point = SVector(1.2*m, 0.0)
        N = 201
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ)

        # Dugdale yield
        addCohesiveConstraint!(problem, DugdaleCohesiveZone(1.0 - r, -r, 1.0, k))

        outputs = [CSVDomainOutput(mesh, "outputs/test_dugdale_$(r)")]

        # Run problem
        run!(problem; outputs=outputs, log=false)

        # Analytical solution
        m_calc = get_crack_ratio(mesh.elems, problem.σ.value, r)

        # Error less than 6%
        @test isapprox(m, m_calc; rtol=6.0e-02)
    end
    @testset "3D normal: r = 0.5" begin
        # Initial shear stress
        r = 0.5
        # Dugdale crack ratio
        m = sqrt(1 - r^2)

        # Create mesh
        mesh = DDMesh2D(Float64, "mesh.msh")

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ, ν=ν)

        # Dugdale yield
        addCohesiveConstraint!(problem, DugdaleCohesiveZone(1.0 - r, -r, 1.0, k))

        outputs = [VTKDomainOutput(mesh, "outputs/test_dugdale_3D_normal_$(r)")]

        # Run problem
        run!(problem; outputs=outputs, log=false)

        # Need to implement way to get m from results
        @test true
    end
    # @testset "3D shear: r = 0.5" begin
    #     # Initial shear stress
    #     r = 0.5
    #     # Dugdale crack ratio
    #     m = 1 / cos(π / 2 * r)

    #     # Create mesh
    #     mesh = DDMesh2D(Float64, "mesh.msh")

    #     # Create problem
    #     problem = ShearDDProblem3D(mesh; μ=μ, ν=ν)

    #     # Dugdale yield
    #     addCohesiveConstraint!(problem, DugdaleCohesiveZone(1.0 - r, -r, 1.0, k))

    #     outputs = [VTKDomainOutput(mesh, "outputs/test_dugdale_3D_shear_$(r)")]

    #     # Run problem
    #     run!(problem; outputs=outputs, log=true)

    #     # # Analytical solution
    #     # m_calc = get_crack_ratio(mesh.elems, problem.σ.value, r)
    #     # # Error
    #     # err = abs(m_calc - m) / m
    #     # # Error less than 3%
    #     # @test err < 0.03
    # end
end
end