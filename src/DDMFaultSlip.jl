module DDMFaultSlip

using StaticArrays
using SparseArrays
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
export ConstantDT
export TimeSequence

include("executioner.jl")

include("variable.jl")

include("constraints/constraint.jl")
export FunctionConstraint

include("constraints/friction.jl")
export ConstantYield
export ConstantFriction
export SlipWeakeningFriction

include("constraints/cohesive_zone.jl")
export DugdaleCohesiveZone

include("fluid_coupling.jl")
export FunctionPressure

include("problem.jl")
export NormalDDProblem, ShearDDProblem
export addNormalDDIC!, addShearDDIC!, addNormalStressIC!, addShearStressIC!
export addConstraint!
export addFrictionConstraint!
export addCohesiveZoneConstraint!
export addFluidCoupling!
export addOutput!

include("jacobian.jl")
export collocation_mul

include("solver.jl")
export DDSolver

include("assembly.jl")

include("outputs.jl")
export VTKDomainOutput
export CSVDomainOutput
export CSVMaximumOutput

# HMatrices.use_global_index() = true

function run!(problem::AbstractDDProblem{T};
    outputs::Vector{<:AbstractOutput}=Vector{AbstractOutput}(undef, 0),
    log::Bool=true, linear_log::Bool=false, output_initial::Bool=false,
    pc::Bool=true, pc_atol::T=1.0e-01,
    nl_max_it::Int64=100, nl_abs_tol::T=1.0e-10, nl_rel_tol::T=1.0e-10,
    l_solver::String="idrs", l_max_it::Int64=1000, l_abs_tol::T=1.0e-10, l_rel_tol::T=1.0e-10,
    hmat_eta::T=2.0, hmat_atol::T=1.0e-06) where {T<:Real}
    # Timer
    timer = TimerOutput()

    # Initialize solver
    @timeit timer "Initialize Solver" solver = DDSolver(problem;
        hmat_eta=hmat_eta, hmat_atol=hmat_atol,
        pc, pc_atol,
        nl_max_it=nl_max_it, nl_abs_tol=nl_abs_tol, nl_rel_tol=nl_rel_tol,
        l_solver=l_solver, l_max_it=l_max_it, l_abs_tol=l_abs_tol, l_rel_tol=l_rel_tol
    )

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
    converged = solve!(solver, problem, timer; log=log, linear_log=linear_log)

    if converged
        # Update "fake" executioner
        advanceTime!(exec)

        # Execute outputs
        if ~isempty(outputs)
            @timeit timer "Execute Outputs" executeOutputs!(outputs, problem, exec, output_initial)
        end
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
    pc::Bool=true, pc_atol::T=1.0e-01,
    nl_max_it::Int64=100, nl_abs_tol::T=1.0e-10, nl_rel_tol::T=1.0e-10,
    l_solver::String="idrs", l_max_it::Int64=1000, l_abs_tol::T=1.0e-10, l_rel_tol::T=1.0e-10,
    hmat_eta::T=2.0, hmat_atol::T=1.0e-06, dt_min::T=1.0e-14) where {T<:Real}
    # Timer
    timer = TimerOutput()

    # Initialize solver
    @timeit timer "Initialize Solver" solver = DDSolver(problem;
        hmat_eta=hmat_eta, hmat_atol=hmat_atol,
        pc, pc_atol,
        nl_max_it=nl_max_it, nl_abs_tol=nl_abs_tol, nl_rel_tol=nl_rel_tol,
        l_solver=l_solver, l_max_it=l_max_it, l_abs_tol=l_abs_tol, l_rel_tol=l_rel_tol
    )

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

    converged = true
    while exec.time < (time_stepper.end_time - time_stepper.tol)
        if converged
            # Update transient executioner
            advanceTime!(exec, time_stepper)

            # Save old state
            @timeit timer "Reinitialize problem" reinit!(problem)
        else
            exec.dt = exec.dt / 2.0
            exec.time = exec.time_old + exec.dt
        end

        # Print time step information
        if log
            print_TimeStepInfo(exec)
        end

        # Pressure update
        computePressureCoupling!(problem, exec.time, timer)
        # Transient problem
        converged = solve!(solver, problem, timer; log=log, linear_log=linear_log)

        if converged
            # Execute outputs
            if ~isempty(outputs)
                @timeit timer "Execute Outputs" executeOutputs!(outputs, problem, exec, output_initial)
            end

            # Reinit solver
            @timeit timer "Reinitialize Solver" reinit!(solver; end_time_step=true)
        end

        if exec.dt < dt_min
            throw(ErrorException("Simulation failed!"))
        end
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
