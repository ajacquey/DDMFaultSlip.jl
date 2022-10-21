function run!(problem::AbstractDDProblem{T}; log::Bool = true, linear_log::Bool = false, nl_max_it::Int64 = 100, nl_abs_tol::T = 1.0e-10, nl_rel_tol::T = 1.0e-10) where {T<:Real}
    # Timer
    timer = TimerOutput()
    
    # Initialize solver
    @timeit timer "Initialize Solver" solver = DDSolver(problem; nl_max_it = nl_max_it, nl_abs_tol = nl_abs_tol, nl_rel_tol = nl_rel_tol)

    # # Initialize outputs
    # if ~isempty(problem.outputs)
    #     @timeit timer "Initialize Outputs" initializeOutputs!(outputs, problem)
    # end

    # Display some information about simulation
    if log
        @timeit timer "Priting simulation information" begin
            show(problem)
            show(solver)
        end
    end

    # Apply ICs
    @timeit timer "Apply Initial Conditions" applyIC!(problem)

    # Steady state problem
    solve!(solver, problem, timer; log = log, linear_log = linear_log)

    # Final update
    # @timeit timer "Final update" final_update!(problem)

    # End of simulation information - TimerOutputs
    if log
        print_timer(timer, title = "Performance graph")
        println()
    end

    return nothing
end

mutable struct TransientExecutioner{T<:Real}
    " Time step number"
    time_step::Int

    " Current time"
    time::T

    " Old time"
    time_old::T

    " Current time step size"
    dt::T

    " Constructor"
    function TransientExecutioner(time_stepper::TimeStepper{T}) where {T<:Real}
        return new{T}(0, time_stepper.start_time, time_stepper.start_time, 0.0)
    end
end
    
function advanceTime!(exec::TransientExecutioner{T}, time_stepper::TimeStepper{T}) where {T<:Real}
    # Update time step
    exec.time_step = exec.time_step + 1
    # Update time old
    exec.time_old = exec.time
    # Update time
    exec.time = get_time(time_stepper, exec.time_step)
    # Update dt
    exec.dt = exec.time - exec.time_old

    return nothing
end

function print_TimeStepInfo(exec::TransientExecutioner{T}) where {T<:Real}
    @printf("\nTime Step %i, time = %.6f, dt = %.6f\n", exec.time_step, exec.time, exec.dt)
    return nothing
end

function run!(problem::AbstractDDProblem{T}, time_stepper::TimeStepper{T}; log::Bool = true, linear_log::Bool = false, nl_max_it::Int64 = 100, nl_abs_tol::T = 1.0e-10, nl_rel_tol::T = 1.0e-10) where {T<:Real}
    # Timer
    timer = TimerOutput()
    
    # Initialize solver
    @timeit timer "Initialize Solver" solver = DDSolver(problem; nl_max_it = nl_max_it, nl_abs_tol = nl_abs_tol, nl_rel_tol = nl_rel_tol)

    # # Initialize outputs
    # if ~isempty(problem.outputs)
    #     @timeit timer "Initialize Outputs" initializeOutputs!(outputs, problem)
    # end

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
        solve!(solver, problem, timer; log = log, linear_log = linear_log)

        # # Final update
        # @timeit timer "Final update" final_update!(problem)

        # # Output
        # if ~isempty(outputs)
        #     @timeit timer "Outputs" outputResults!(outputs, problem)
        # end

        # Reinit solver
        @timeit timer "Reinitialize Solver" reinit!(solver; end_time_step = true)
    end

    # Final update
    # @timeit timer "Final update" final_update!(problem)

    # End of simulation information - TimerOutputs
    if log
        print_timer(timer, title = "Performance graph")
        println()
    end

    return nothing
end