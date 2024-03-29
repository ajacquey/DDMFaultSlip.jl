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
h = 1.0e-05
k = μ / h

function get_crack_ratio(elems, σ, λ)
    idx = findall(x -> x >= 0.99 * (1.0 - λ), σ)
    if length(idx) > 0
        return 1 / elems[idx[end]].X[1]
    else
        return NaN
    end
end

function get_crack_ratio(elems, w)
    idx = findall(x -> x >= 1.0e-05, w)
    r = [norm(elems[k].X) for k in eachindex(elems)]

    if length(idx) > 0
        return 1 / maximum(r[idx])
    else
        return NaN
    end
end

@testset "Dugdale cohezive zone" begin
    @testset "2D: r = 0.25" begin
        # Stress ratio
        λ = 0.25
        # Dugdale crack ratio
        c = cos(π / 2 * λ)

        # Create mesh
        start_point = SVector(-1.2 / c, 0.0)
        end_point = SVector(1.2 / c, 0.0)
        N = 201
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ)

        # Dugdale yield
        addCohesiveZoneConstraint!(problem, DugdaleCohesiveZone(1.0 - λ, -λ, 1.0, k))
        
        # Run problem
        run!(problem; log=false)

        # Analytical solution
        c_calc = get_crack_ratio(mesh.elems, problem.σ.value, λ)

        # Error less than 3%
        @test isapprox(c_calc, c; rtol=3.0e-02)
    end
    @testset "2D: r = 0.5" begin
        # Stress ratio
        λ = 0.5
        # Dugdale crack ratio
        c = cos(π / 2 * λ)

        # Create mesh
        start_point = SVector(-1.2 / c, 0.0)
        end_point = SVector(1.2 / c, 0.0)
        N = 201
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ)

        # Dugdale yield
        addCohesiveZoneConstraint!(problem, DugdaleCohesiveZone(1.0 - λ, -λ, 1.0, k))

        # Run problem
        run!(problem; log=false)

        # Analytical solution
        c_calc = get_crack_ratio(mesh.elems, problem.σ.value, λ)

        # Error less than 3%
        @test isapprox(c_calc, c; rtol=3.0e-02)
    end
    @testset "2D: r = 0.75" begin
        # Stress ratio
        λ = 0.75
        # Dugdale crack ratio
        c = cos(π / 2 * λ)

        # Create mesh
        start_point = SVector(-1.2 / c, 0.0)
        end_point = SVector(1.2 / c, 0.0)
        N = 201
        mesh = DDMesh1D(start_point, end_point, N)

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ)

        # Dugdale yield
        addCohesiveZoneConstraint!(problem, DugdaleCohesiveZone(1.0 - λ, -λ, 1.0, k))

        # Run problem
        run!(problem; log=false)

        # Analytical solution
        c_calc = get_crack_ratio(mesh.elems, problem.σ.value, λ)

        # Error less than 6%
        @test isapprox(c, c_calc; rtol=6.0e-02)
    end
    @testset "3D normal: r = 0.5" begin
        # Stress ratio
        λ = 0.5
        # Dugdale crack ratio
        c = sqrt(1 - λ^2)
        # Poisson's ratio
        ν = 0.25
        # Create mesh
        mesh = DDMesh2D(Float64, "mesh-dugdale.msh")

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ, ν=ν)

        # Dugdale yield
        addCohesiveZoneConstraint!(problem, DugdaleCohesiveZone(1.0 - λ, -λ, 1.0, k))

        # Run problem
        run!(problem; log=false)

        # Analytical solution
        c_calc = get_crack_ratio(mesh.elems, problem.w.value)

        # Need to implement way to get c from results
        @test isapprox(c, c_calc; rtol=2.0e-02)
    end
    @testset "3D shear: ν = 0, r = 0.5" begin
        # Stress ratio
        λ = 0.5
        # Dugdale crack ratio
        c = sqrt(1 - λ^2)
        # Poisson's ratio
        ν = 0.0

        function τ₀(X)
            return λ * ones(length(X))
        end

        # Create mesh
        mesh = DDMesh2D(Float64, "mesh-dugdale.msh")

        # Create problem
        problem = ShearDDProblem(mesh; μ=μ, ν=ν)

        # Initial shear stress
        addShearStressIC!(problem, τ₀)

        # Dugdale yield
        addCohesiveZoneConstraint!(problem, DugdaleCohesiveZone(1.0 - λ, -λ, 1.0, k))

        # Run problem
        run!(problem; log=false)

        # Need to implement way to get c from results
        @test true
    end
    @testset "3D shear: ν = 0.25, r = 0.5" begin
        # Stress ratio
        λ = 0.5
        # Dugdale crack ratio
        c = sqrt(1 - λ^2)
        # Poisson's ratio
        ν = 0.25

        function τ₀(X, sym)
            if (sym == :x)
                return λ * ones(length(X))
            else
                return zeros(length(X))
            end
        end

        # Create mesh
        mesh = DDMesh2D(Float64, "mesh-dugdale.msh")

        # Create problem
        problem = ShearDDProblem(mesh; μ=μ, ν=ν)

        # Initial shear stress
        addShearStressIC!(problem, τ₀)

        # Dugdale yield
        addCohesiveZoneConstraint!(problem, DugdaleCohesiveZone(1.0 - λ, -λ, 1.0, k))

        # Run problem
        run!(problem; log=false)

        # Need to implement way to get c from results
        @test true
    end
end
end