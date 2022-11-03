module Outputs

using DDMFaultSlip
using StaticArrays
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
        outputs = [VTKDomainOutput(mesh, "outputs/test_vtk_3d_opening"), CSVMaximumOutput("outputs/test_vtk_3d_opening")]

        # Run problem
        run!(problem; outputs=outputs, log=false)

        # Test if output files exist
        @test (isfile("outputs/test_vtk_3d_opening.pvd") && isfile("outputs/test_vtk_3d_opening_1.vtu") && isfile("outputs/test_vtk_3d_opening.csv"))
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
        outputs = [VTKDomainOutput(mesh, "outputs/test_vtk_3d_opening_initial"), CSVMaximumOutput("outputs/test_vtk_3d_opening_initial")]

        # Run problem
        run!(problem; outputs=outputs, output_initial=true, log=false)

        # Test if output files exist
        @test (isfile("outputs/test_vtk_3d_opening_initial.pvd") && isfile("outputs/test_vtk_3d_opening_initial_0.vtu") && isfile("outputs/test_vtk_3d_opening_initial_1.vtu") && isfile("outputs/test_vtk_3d_opening_initial.csv"))
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
        outputs = [VTKDomainOutput(mesh, "outputs/test_vtk_3d_shear"), CSVMaximumOutput("outputs/test_vtk_3d_shear")]

        # Run problem
        run!(problem; outputs=outputs, log=false)

        # Test if output files exist
        @test (isfile("outputs/test_vtk_3d_shear.pvd") && isfile("outputs/test_vtk_3d_shear_1.vtu") && isfile("outputs/test_vtk_3d_shear.csv"))
    end
end
@testset "CSV output" begin
    @testset "NormalDDProblem - no initial output" begin
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

        # Outputs
        outputs = [CSVDomainOutput(mesh, "outputs/test_csv_2d_opening"), CSVMaximumOutput("outputs/test_csv_2d_opening")]

        # Run problem
        run!(problem; outputs=outputs, log=false)

        # Test if output files exist
        @test (isfile("outputs/test_csv_2d_opening_1.csv") && isfile("outputs/test_csv_2d_opening.csv"))
    end
    @testset "NormalDDProblem - with initial output" begin
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

        # Outputs
        outputs = [CSVDomainOutput(mesh, "outputs/test_csv_2d_opening_initial"), CSVMaximumOutput("outputs/test_csv_2d_opening_initial")]

        # Run problem
        run!(problem; outputs=outputs, output_initial=true, log=false)

        # Test if output files exist
        @test (isfile("outputs/test_csv_2d_opening_initial_0.csv") && isfile("outputs/test_csv_2d_opening_initial_1.csv") && isfile("outputs/test_csv_2d_opening_initial.csv"))
    end
    @testset "ShearDDProblem - no initial output" begin
        # Create mesh
        start_point = SVector(-1.0, 0.0)
        end_point = SVector(1.0, 0.0)
        N = 100
        mesh = DDMesh1D(start_point, end_point, N)

        # Elastic property
        μ = 1.0

        # Create problem
        problem = ShearDDProblem2D(mesh; μ=μ)
        addConstraint!(problem, FunctionConstraint(σ_cst))

        # Outputs
        outputs = [CSVDomainOutput(mesh, "outputs/test_csv_2d_shear"), CSVMaximumOutput("outputs/test_csv_2d_shear")]

        # Run problem
        run!(problem; outputs=outputs, log=false)

        # Test if output files exist
        @test (isfile("outputs/test_csv_2d_shear_1.csv") && isfile("outputs/test_csv_2d_shear.csv"))
    end
end
end