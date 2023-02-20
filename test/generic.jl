module TestGeneric

using DDMFaultSlip
using StaticArrays
using Statistics
using Test

function σ_cst(X, time::T) where {T<:Real}
    return 1.5 * (2.0 * X[1]^2 - 1)
end

function ϵ_analytical(mesh::DDMesh1D{T}, μ::T)::Vector{T} where {T<:Real}
    x = [mesh.elems[i].X[1] for i in 1:length(mesh.elems)]

    return (1.0 .- x .^ 2) .^ 1.5
end

@testset "Generic stress" begin
    @testset "PWC 2D" begin
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
        run!(problem; log=false)

        # Analytical solution
        ϵ_sol = ϵ_analytical(mesh, μ)
        # Error
        # err = mean(abs.(problem.ϵ.value - ϵ_sol) ./ ϵ_sol)
        # Error less than 2%
        # @test err < 0.02
        @test isapprox(problem.ϵ.value, ϵ_sol; rtol=2.0e-02)
    end
end

end