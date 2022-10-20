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

function DD_analytical(mesh::DDMesh1D{T}, μ::T)::Vector{T} where {T<:Real}
    x = [mesh.elems[i].X[1] for i in 1:length(mesh.elems)]

    return sqrt.(1.0 .- x.^2) / μ
end

@testset "Coupled problems" begin
    @testset "Fake coupling" begin
        # Create mesh
        start_point = SVector(-1.0, 0.0)
        end_point = SVector(1.0, 0.0)
        N = 100
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = CoupledDDProblem2D(mesh; μ = μ)
        addConstraint!(problem, :ϵ, FunctionConstraint(σ_cst))
        addConstraint!(problem, :δ, FunctionConstraint(τ_cst))

        # Run problem
        run!(problem; log = false)

        # Analytical solutions
        DD_sol = DD_analytical(mesh, μ)
        # Error
        err = mean(vcat(abs.(problem.ϵ.value - DD_sol) ./ DD_sol, abs.(problem.δ.value - DD_sol) ./ DD_sol)) 
        # Error less than 2%
        @test err < 0.02
    end
end

end