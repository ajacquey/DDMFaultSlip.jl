module Outputs

using DDMFaultSlip
using Test

function σ_cst(X, time::T) where {T<:Real}
    return -1.0
end

@testset "VTK output" begin
    @testset "NormalDDProblem - no initial output" begin
        # Create mesh
        mesh = DDMesh2D(Float64, "mesh.msh")

        # Elastic property
        μ = 1.0
        ν = 0.0

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ, ν=ν)
        addConstraint!(problem, FunctionConstraint(σ_cst))

        # Outputs
        outputs = [VTKDomainOutput(mesh, "outputs/test_vtk_3d_opening")]

        # Run problem
        run!(problem; outputs=outputs, log=false)

        # Test if output files exist
        @test (isfile("outputs/test_vtk_3d_opening.pvd") && isfile("outputs/test_vtk_3d_opening_1.vtu"))
    end
    @testset "NormalDDProblem - with initial output" begin
        # Create mesh
        mesh = DDMesh2D(Float64, "mesh.msh")

        # Elastic property
        μ = 1.0
        ν = 0.0

        # Create problem
        problem = NormalDDProblem(mesh; μ=μ, ν=ν)
        addConstraint!(problem, FunctionConstraint(σ_cst))

        # Outputs
        outputs = [VTKDomainOutput(mesh, "outputs/test_vtk_3d_opening_initial")]

        # Run problem
        run!(problem; outputs=outputs, output_initial=true, log=false)

        # Test if output files exist
        @test (isfile("outputs/test_vtk_3d_opening_initial.pvd") && isfile("outputs/test_vtk_3d_opening_initial_0.vtu") && isfile("outputs/test_vtk_3d_opening_initial_1.vtu"))
    end
    @testset "ShearDDProblem - no initial output" begin
        # Create mesh
        mesh = DDMesh2D(Float64, "mesh.msh")

        # Elastic property
        μ = 1.0
        ν = 0.0

        # Create problem
        problem = ShearDDProblem3D(mesh; μ=μ, ν=ν)
        addConstraint!(problem, :x, FunctionConstraint(σ_cst))

        # Outputs
        outputs = [VTKDomainOutput(mesh, "outputs/test_vtk_3d_shear")]

        # Run problem
        run!(problem; outputs=outputs, log=false)

        # Test if output files exist
        @test (isfile("outputs/test_vtk_3d_shear.pvd") && isfile("outputs/test_vtk_3d_shear_1.vtu"))
    end
end
end