module TestCoupled

using DDMFaultSlip
using StaticArrays
using Statistics
using Test

function σ_cst(X, time::T) where {T<:Real}
    return -1.0
end

function τ_cst(X, time::T) where {T<:Real}
    return -1.0
end

function ϵ_analytical_2D(mesh::DDMesh1D{T}, μ::T)::Vector{T} where {T<:Real}
    x = [mesh.elems[i].X[1] for i in 1:length(mesh.elems)]

    return sqrt.(1.0 .- x .^ 2) / μ
end

function ϵ_analytical_3D(mesh::DDMesh2D{T}, μ::T, ν::T)::Vector{T} where {T<:Real}
    r = [sqrt(mesh.elems[i].X[1]^2 + mesh.elems[i].X[2]^2) for i in 1:length(mesh.elems)]

    return 4 * (1 - ν) / (π * μ) * sqrt.(1 .- r .^ 2)
end

function δ_analytical_2D(mesh::DDMesh1D{T}, μ::T)::Vector{T} where {T<:Real}
    x = [mesh.elems[i].X[1] for i in 1:length(mesh.elems)]

    return sqrt.(1.0 .- x .^ 2) / μ
end

function δ_analytical_3D(mesh::DDMesh2D{T}, μ::T, ν::T)::Vector{T} where {T<:Real}
    r = [sqrt(mesh.elems[i].X[1]^2 + mesh.elems[i].X[2]^2) for i in 1:length(mesh.elems)]

    return 4 * (1 - ν) / (π * (1 - ν / 2) * μ) * sqrt.(1 .- r .^ 2)
end

@testset "Coupled problems" begin
    @testset "Fake coupling 2D" begin
        # Create mesh
        start_point = SVector(-1.0, 0.0)
        end_point = SVector(1.0, 0.0)
        N = 100
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = CoupledDDProblem2D(mesh; μ=μ)
        addConstraint!(problem, :ϵ, FunctionConstraint(σ_cst))
        addConstraint!(problem, :δ, FunctionConstraint(τ_cst))

        # Run problem
        run!(problem; log=false)

        # Analytical solutions
        ϵ_sol = ϵ_analytical_2D(mesh, μ)
        δ_sol = δ_analytical_2D(mesh, μ)

        # Error less than 2%
        @test isapprox(problem.ϵ.value, ϵ_sol; rtol=2.0e-02) && isapprox(problem.δ.value, δ_sol; rtol=2.0e-02)
    end
    @testset "Fake coupling 3D - x" begin
        # Create mesh
        mesh = DDMesh2D(Float64, "mesh.msh")

        # Elastic property
        μ = 1.0
        ν = 0.0

        # Create problem
        problem = CoupledDDProblem3D(mesh; μ=μ, ν=ν)
        addConstraint!(problem, :ϵ, FunctionConstraint(σ_cst))
        addConstraint!(problem, :δ_x, FunctionConstraint(τ_cst))

        # Run problem
        run!(problem; log=false)

        # Analytical solutions
        ϵ_sol = ϵ_analytical_3D(mesh, μ, ν)
        δ_sol = δ_analytical_3D(mesh, μ, ν)

        # Error less than 4%
        @test isapprox(problem.ϵ.value, ϵ_sol; rtol=4.0e-02) && isapprox(problem.δ_x.value, δ_sol; rtol=4.0e-02)
    end
    @testset "Fake coupling 3D - y" begin
        # Create mesh
        mesh = DDMesh2D(Float64, "mesh.msh")

        # Elastic property
        μ = 1.0
        ν = 0.0

        # Create problem
        problem = CoupledDDProblem3D(mesh; μ=μ, ν=ν)
        addConstraint!(problem, :ϵ, FunctionConstraint(σ_cst))
        addConstraint!(problem, :δ_y, FunctionConstraint(τ_cst))

        # Run problem
        run!(problem; log=false)

        # Analytical solutions
        ϵ_sol = ϵ_analytical_3D(mesh, μ, ν)
        δ_sol = δ_analytical_3D(mesh, μ, ν)

        # Error less than 4%
        @test isapprox(problem.ϵ.value, ϵ_sol; rtol=4.0e-02) && isapprox(problem.δ_y.value, δ_sol; rtol=4.0e-02)
    end
end

end