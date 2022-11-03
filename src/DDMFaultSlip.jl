module DDMFaultSlip

using StaticArrays
using LinearAlgebra
using Statistics: mean
using InteractiveUtils
using Printf
using HMatrices
using IterativeSolvers
using TimerOutputs
using WriteVTK
using DelimitedFiles

include("mesh.jl")
export Point2D, Point3D, DDMesh1D, DDMesh2D

include("collocation.jl")
export DD3DShearElasticMatrix

include("time_stepper.jl")
export TimeSequence

include("executioner.jl")

include("variable.jl")

include("constraints/constraint.jl")
export FunctionConstraint

include("constraints/friction.jl")
export ConstantYield
export ConstantFriction

include("fluid_coupling.jl")
export FunctionPressure

include("problem.jl")
export NormalDDProblem, ShearDDProblem2D, ShearDDProblem3D, CoupledDDProblem2D, CoupledDDProblem3D
export addNormalStressIC!, addShearStressIC!
export addConstraint!
export addFrictionConstraint!
export addFluidCoupling!
export addOutput!

include("jacobian.jl")

include("solver.jl")

include("assembly.jl")

include("outputs.jl")
export VTKDomainOutput
export CSVDomainOutput
export CSVMaximumOutput

function run!(problem::AbstractDDProblem{T};
    outputs::Vector{<:AbstractOutput}=Vector{AbstractOutput}(undef, 0),
    log::Bool=true, linear_log::Bool=false, output_initial::Bool=false,
    nl_max_it::Int64=100, nl_abs_tol::T=1.0e-10, nl_rel_tol::T=1.0e-10) where {T<:Real}
    # Timer
    timer = TimerOutput()

    # Initialize solver
    @timeit timer "Initialize Solver" solver = DDSolver(problem; nl_max_it=nl_max_it, nl_abs_tol=nl_abs_tol, nl_rel_tol=nl_rel_tol)

    # Display some information about simulation
    if log
        @timeit timer "Priting simulation information" begin
            show(problem)
            show(solver)
        end
    end

    # Apply ICs
    @timeit timer "Apply Initial Conditions" applyIC!(problem)

    # Steady-state problem
    exec = SteadyExecutioner(T)

    # Initialize outputs
    if ~isempty(outputs)
        @timeit timer "Initialize Outputs" initializeOutputs!(outputs, problem, exec, output_initial)
    end

    # Steady state problem
    solve!(solver, problem, timer; log=log, linear_log=linear_log)

    # Update "fake" executioner
    advanceTime!(exec)

    # Execute outputs
    if ~isempty(outputs)
        @timeit timer "Execute Outputs" executeOutputs!(outputs, problem, exec, output_initial)
    end

    # End of simulation information - TimerOutputs
    if log
        print_timer(timer, title="Performance graph")
        println()
    end

    return nothing
end

function run!(problem::AbstractDDProblem{T}, time_stepper::AbstractTimeStepper{T};
    outputs::Vector{<:AbstractOutput}=Vector{AbstractOutput}(undef, 0),
    log::Bool=true, linear_log::Bool=false, output_initial::Bool=false,
    nl_max_it::Int64=100, nl_abs_tol::T=1.0e-10, nl_rel_tol::T=1.0e-10) where {T<:Real}
    # Timer
    timer = TimerOutput()

    # Initialize solver
    @timeit timer "Initialize Solver" solver = DDSolver(problem; nl_max_it=nl_max_it, nl_abs_tol=nl_abs_tol, nl_rel_tol=nl_rel_tol)

    # Display some information about simulation
    if log
        @timeit timer "Priting simulation information" begin
            show(problem)
            show(solver)
        end
    end

    # Apply ICs
    @timeit timer "Apply Initial Conditions" applyIC!(problem)

    # Transient problem
    exec = TransientExecutioner(time_stepper)
    if log
        print_TimeStepInfo(exec)
    end

    # Initialize outputs
    if ~isempty(outputs)
        @timeit timer "Initialize Outputs" initializeOutputs!(outputs, problem, exec, output_initial)
    end

    while exec.time < (time_stepper.end_time - time_stepper.tol)
        # Update transient executioner
        advanceTime!(exec, time_stepper)

        # Save old state
        @timeit timer "Reinitialize problem" reinit!(problem)

        # Print time step information
        if log
            print_TimeStepInfo(exec)
        end

        # Pressure update
        computePressureCoupling!(problem, exec.time, timer)
        # Transient problem
        solve!(solver, problem, timer; log=log, linear_log=linear_log)

        # Execute outputs
        if ~isempty(outputs)
            @timeit timer "Execute Outputs" executeOutputs!(outputs, problem, exec, output_initial)
        end

        # Reinit solver
        @timeit timer "Reinitialize Solver" reinit!(solver; end_time_step=true)
    end

    # End of simulation information - TimerOutputs
    if log
        print_timer(timer, title="Performance graph")
        println()
    end

    return nothing
end

export run!
end
